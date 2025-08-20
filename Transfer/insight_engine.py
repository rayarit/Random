# src/insights/insight_engine.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import math

# ---------- Formatting helpers ----------
SCALE_THRESHOLDS = [
    (1_000_000_000, "B"),
    (1_000_000, "M"),
    (1_000, "K"),
]

def _humanize_number(x: Optional[float]) -> str:
    try:
        val = float(x)
    except Exception:
        return "N/A"
    sign = "-" if val < 0 else ""
    val = abs(val)
    for thr, sym in SCALE_THRESHOLDS:
        if val >= thr:
            return f"{sign}{val / thr:.1f}{sym}"
    return f"{sign}{val:.1f}"

def _fmt_money(x: Optional[float]) -> str:
    h = _humanize_number(x)
    return f"${h}" if h != "N/A" else h

def _fmt_pct(x: Optional[float]) -> str:
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    try:
        n = float(n); d = float(d)
        if d == 0: return None
        return n / d
    except Exception:
        return None

# ---------- Column detection ----------
DATE_COLS = ["ORDER_DATE", "DS", "order_date", "ds"]
TY_FLAG_COLS = ["YEAR_FLAG", "FY_FLAG", "year_flag", "fy_flag"]
TY_FLAG_TY = {"TY", "This Year", "ty", "this_year"}
TY_FLAG_LY = {"LY", "Last Year", "ly", "last_year"}

METRIC_COLS = [
    "DEMAND_SALES", "DEMAND_QTY",
    "TREND_SALES", "TREND_QTY",
    "TOTAL_VISITS", "A2C_VISITS", "CART_VISITS", "CHECKOUT_VISITS", "PURCHASE_VISITS",
    "DEMAND_ORDERS"
]

DIM_CANDIDATES = ["PLATFORM", "CHANNEL", "GMM", "DMM", "CATEGORY", "MEMBER_TYPE", "MEMBERSHIP_TYPE", "TENURE_FLAG", "RESELLER_FLAG"]

def _first_present(cols: List[str], df_cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df_cols:
            return c
    return None

def _present(df: pd.DataFrame, names: List[str]) -> List[str]:
    return [c for c in names if c in df.columns]

# ---------- Core compute ----------
def compute_insights(df: pd.DataFrame, question: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Returns (facts_dict, highlights) for SummaryAgent + UI.
    Facts are already human-formatted (K/M/B, $, %).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ({
            "period": "N/A",
            "kpis": {},
            "top_contributors_pos": [],
            "top_contributors_neg": [],
            "funnel": {},
            "anomalies": [],
            "caveats": ["No data returned for the requested filters/period."]
        }, [{"label": "Status", "value": "No data"}])

    df = df.copy()
    cols = list(df.columns)

    # Detect columns
    date_col = _first_present(DATE_COLS, cols)
    ty_col = _first_present(TY_FLAG_COLS, cols)

    # Metrics present
    metrics_present = _present(df, METRIC_COLS)

    # Period
    period = "N/A"
    if date_col:
        try:
            # coerce to datetime for min/max if possible
            df["_tmp_date"] = pd.to_datetime(df[date_col], errors="coerce")
            dmin = df["_tmp_date"].min()
            dmax = df["_tmp_date"].max()
            if pd.notna(dmin) and pd.notna(dmax):
                period = f"{dmin.date()} to {dmax.date()}"
            df.drop(columns=["_tmp_date"], inplace=True, errors="ignore")
        except Exception:
            pass

    # Aggregate to overall totals
    agg_map = {m: "sum" for m in metrics_present}
    totals = df.agg(agg_map) if agg_map else pd.Series(dtype=float)

    # Split TY/LY totals if TY flag exists
    ty_totals = pd.Series(dtype=float)
    ly_totals = pd.Series(dtype=float)
    if ty_col:
        ty_mask = df[ty_col].astype(str).isin(TY_FLAG_TY)
        ly_mask = df[ty_col].astype(str).isin(TY_FLAG_LY)
        if agg_map:
            ty_totals = df[ty_mask].agg(agg_map)
            ly_totals = df[ly_mask].agg(agg_map)

    # Compute KPIs
    kpis: Dict[str, Dict[str, Any]] = {}

    def add_yoy_metric(name: str, money=False):
        ty_val = float(ty_totals.get(name, float("nan"))) if not ty_totals.empty else float("nan")
        ly_val = float(ly_totals.get(name, float("nan"))) if not ly_totals.empty else float("nan")
        yoy = _safe_div((ty_val - ly_val), ly_val)
        pretty_ty = _fmt_money(ty_val) if money else _humanize_number(ty_val)
        pretty_ly = _fmt_money(ly_val) if money else _humanize_number(ly_val)
        pretty_yoy = _fmt_pct(100.0 * yoy) if yoy is not None else "N/A"
        kpis[name.lower()] = {"TY": pretty_ty, "LY": pretty_ly, "YoY_pct": pretty_yoy}

    if "DEMAND_SALES" in metrics_present:
        add_yoy_metric("DEMAND_SALES", money=True)
    if "DEMAND_QTY" in metrics_present:
        add_yoy_metric("DEMAND_QTY")
    if "TREND_SALES" in metrics_present:
        add_yoy_metric("TREND_SALES", money=True)
    if "TREND_QTY" in metrics_present:
        add_yoy_metric("TREND_QTY")

    # Conversion, AOV, AUR (TY/LY if possible)
    def ty_ly_ratio(num_col, den_col, name):
        ty_num = float(ty_totals.get(num_col, float("nan"))) if not ty_totals.empty else None
        ty_den = float(ty_totals.get(den_col, float("nan"))) if not ty_totals.empty else None
        ly_num = float(ly_totals.get(num_col, float("nan"))) if not ly_totals.empty else None
        ly_den = float(ly_totals.get(den_col, float("nan"))) if not ly_totals.empty else None
        ty_ratio = _safe_div(ty_num, ty_den)
        ly_ratio = _safe_div(ly_num, ly_den)
        yoy = _safe_div((ty_ratio - ly_ratio), ly_ratio) if (ty_ratio is not None and ly_ratio is not None) else None
        if name.lower() == "conversion":
            fmt = _fmt_pct
            scale = 100.0
            ty_val_pretty = fmt(ty_ratio * scale) if ty_ratio is not None else "N/A"
            ly_val_pretty = fmt(ly_ratio * scale) if ly_ratio is not None else "N/A"
        elif name.lower() == "aov":
            ty_val_pretty = _fmt_money(ty_ratio) if ty_ratio is not None else "N/A"
            ly_val_pretty = _fmt_money(ly_ratio) if ly_ratio is not None else "N/A"
        else:  # AUR
            ty_val_pretty = _fmt_money(ty_ratio) if ty_ratio is not None else "N/A"
            ly_val_pretty = _fmt_money(ly_ratio) if ly_ratio is not None else "N/A"
        yoy_pretty = _fmt_pct(100.0 * yoy) if yoy is not None else "N/A"
        kpis[name.lower()] = {"TY": ty_val_pretty, "LY": ly_val_pretty, "YoY_pct": yoy_pretty}

    if {"DEMAND_ORDERS", "TOTAL_VISITS"}.issubset(metrics_present):
        ty_ly_ratio("DEMAND_ORDERS", "TOTAL_VISITS", "conversion")
    if {"DEMAND_SALES", "DEMAND_ORDERS"}.issubset(metrics_present):
        ty_ly_ratio("DEMAND_SALES", "DEMAND_ORDERS", "AOV")
    if {"DEMAND_SALES", "DEMAND_QTY"}.issubset(metrics_present):
        ty_ly_ratio("DEMAND_SALES", "DEMAND_QTY", "AUR")

    # Grain detection & top contributors
    grain: List[str] = []
    top_pos: List[Dict[str, Any]] = []
    top_neg: List[Dict[str, Any]] = []
    # choose one preferred dim if many exist
    preferred_order = ["PLATFORM", "CHANNEL", "GMM", "DMM", "CATEGORY"]
    group_dim = next((d for d in preferred_order if d in cols), None)
    if group_dim and any(m in metrics_present for m in ["DEMAND_SALES", "DEMAND_QTY", "DEMAND_ORDERS", "TOTAL_VISITS"]):
        grain = [group_dim.lower()]
        # aggregate TY + LY by group
        df_ty = df[df[ty_col].astype(str).isin(TY_FLAG_TY)] if ty_col else df
        df_ly = df[df[ty_col].astype(str).isin(TY_FLAG_LY)] if ty_col else pd.DataFrame(columns=df.columns)
        def _agg_by(d):
            if d.empty: 
                return pd.DataFrame(columns=[group_dim, "DEMAND_SALES", "DEMAND_QTY", "DEMAND_ORDERS", "TOTAL_VISITS"]).assign(**{group_dim: []})
            return d.groupby(group_dim, dropna=False)[_present(d, ["DEMAND_SALES","DEMAND_QTY","DEMAND_ORDERS","TOTAL_VISITS"])].sum().reset_index()
        g_ty = _agg_by(df_ty).rename(columns={c: f"{c}_TY" for c in g_ty.columns if c != group_dim}) if not df_ty.empty else pd.DataFrame()
        g_ly = _agg_by(df_ly).rename(columns={c: f"{c}_LY" for c in g_ly.columns if c != group_dim}) if not df_ly.empty else pd.DataFrame()
        g = pd.merge(g_ty, g_ly, on=group_dim, how="left") if not g_ty.empty else g_ly

        target = "DEMAND_SALES"
        if target not in metrics_present:
            target = next((m for m in ["DEMAND_ORDERS", "DEMAND_QTY", "TOTAL_VISITS"] if m in metrics_present), None)

        if target and not g.empty:
            g["_delta"] = g.get(f"{target}_TY", 0.0) - g.get(f"{target}_LY", 0.0)
            # sort
            pos = g.sort_values("_delta", ascending=False).head(3)
            neg = g.sort_values("_delta", ascending=True).head(3)
            total_delta = g["_delta"].sum() or 0.0
            def _row_to_item(row):
                share = _safe_div(row["_delta"], total_delta)
                return {
                    "segment": str(row[group_dim]),
                    "metric": target.lower(),
                    "delta": _fmt_money(row["_delta"]) if "SALES" in target else _humanize_number(row["_delta"]),
                    "share_of_delta": _fmt_pct(100.0 * share) if share is not None else "N/A"
                }
            top_pos = [ _row_to_item(r) for _, r in pos.iterrows() if r["_delta"] > 0 ]
            top_neg = [ _row_to_item(r) for _, r in neg.iterrows() if r["_delta"] < 0 ]

    # Funnel (if present)
    funnel = {}
    funnel_cols = _present(df, ["TOTAL_VISITS","A2C_VISITS","CART_VISITS","CHECKOUT_VISITS","PURCHASE_VISITS"])
    if funnel_cols:
        # Use TY slice if available
        base = df[df[ty_col].astype(str).isin(TY_FLAG_TY)] if ty_col else df
        sums = base[funnel_cols].sum()
        def fmt(c): return _humanize_number(sums.get(c, 0.0))
        funnel = {
            "visits": fmt("TOTAL_VISITS") if "TOTAL_VISITS" in funnel_cols else None,
            "a2c": fmt("A2C_VISITS") if "A2C_VISITS" in funnel_cols else None,
            "cart": fmt("CART_VISITS") if "CART_VISITS" in funnel_cols else None,
            "checkout": fmt("CHECKOUT_VISITS") if "CHECKOUT_VISITS" in funnel_cols else None,
            "purchase": fmt("PURCHASE_VISITS") if "PURCHASE_VISITS" in funnel_cols else None,
        }
        # biggest drop
        steps = [("visits","a2c"), ("a2c","cart"), ("cart","checkout"), ("checkout","purchase")]
        biggest = None; biggest_val = -1.0
        # use raw numbers for drop computation
        v_raw = base[funnel_cols].sum()
        def raw(c): 
            try: return float(v_raw.get(c, 0.0))
            except Exception: return 0.0
        for s, t in steps:
            if s.upper()+"_VISITS" in funnel_cols and t.upper()+"_VISITS" in funnel_cols:
                drop = _safe_div((raw(s.upper()+"_VISITS") - raw(t.upper()+"_VISITS")), raw(s.upper()+"_VISITS"))
                if drop is not None and drop > biggest_val:
                    biggest_val = drop
                    biggest = (s, t, drop)
        if biggest is not None:
            funnel["biggest_drop"] = {"step": f"{biggest[0].capitalize()} → {biggest[1].capitalize()}", "drop_pct": _fmt_pct(100.0 * biggest[2])}

    # Caveats
    caveats = []
    # If query tried to roll traffic to GMM/DMM/etc., warn (we can only infer based on available columns)
    if any(c in cols for c in ["GMM","DMM","CATEGORY","TENURE_FLAG","MEMBERSHIP_TYPE","RESELLER_FLAG"]):
        # but only warn if traffic columns are present
        if any(c in metrics_present for c in ["TOTAL_VISITS","A2C_VISITS","CART_VISITS","CHECKOUT_VISITS","PURCHASE_VISITS"]):
            caveats.append("Traffic metrics are rolled to date/platform; cannot be reliably aggregated to GMM/DMM/Category/Tenure/Membership Type/Reseller.")

    facts: Dict[str, Any] = {
        "period": period,
        "grain": grain,
        "kpis": kpis,
        "top_contributors_pos": top_pos,
        "top_contributors_neg": top_neg,
        "funnel": funnel,
        "anomalies": [],   # reserved for V2
        "caveats": caveats
    }

    # Highlights for UI cards (best-effort)
    highlights: List[Dict[str, str]] = []
    if "demand_sales" in kpis:
        highlights.append({"label": "YoY Sales", "value": f"{kpis['demand_sales']['TY']} ({kpis['demand_sales']['YoY_pct']})"})
    if "conversion" in kpis:
        highlights.append({"label": "Conversion", "value": f"{kpis['conversion']['TY']} ({kpis['conversion']['YoY_pct']})"})
    if "AOV".lower() in kpis:
        highlights.append({"label": "AOV", "value": f"{kpis['aov']['TY']} ({kpis['aov']['YoY_pct']})"})
    if "top_contributors_pos" in facts and facts["top_contributors_pos"]:
        tp = facts["top_contributors_pos"][0]
        highlights.append({"label": "Top Driver", "value": f"{tp['segment']} ({tp['delta']}, {tp['share_of_delta']})"})
    if funnel and "biggest_drop" in funnel:
        bd = funnel["biggest_drop"]
        highlights.append({"label": "Biggest Funnel Drop", "value": f"{bd['step']} ({bd['drop_pct']})"})

    return facts, highlights
