system_message = (
        "You are a BigQuery SQL expert. Write a SQL query that answers the user's question "
        "using ONLY the exact table and column names provided below. "
        "DO NOT invent columns or tables.\n"
        "Rules:\n"
        "- Use fully qualified tables with project+dataset in backticks, e.g., "
        f"`{self.project_id}.dataset.table_name`.\n"
        "- Output ONLY the SQL statement. No explanations, no code fences.\n"
        "- Return RAW numeric values (no K/M/B, %, '$', string concat, or CAST to string).\n"
        "- Give aggregates a clear alias (e.g., `... AS demand_sales_num`) and ORDER BY that alias.\n"
        "- Do NOT perform aggregations of aggregations. If needed, aggregate in a CTE/subquery, "
        "then select from it.\n"
        "- Keep to valid BigQuery SQL syntax."
    )

##==================== llm.invoke()
def _extract_sql_text(self, text: str) -> str:
    import re
    if not isinstance(text, str):
        text = str(text)
    # remove ```sql ... ``` or ``` ... ```
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # remove a leading 'sql' token on its own line
    text = re.sub(r"^\s*sql\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


##=====================================

## Update 1 
# --- NEW robust SQL extraction + gating ---
raw_text = str(llm_response).strip()

import re

def _extract_sql_from_text(text: str) -> str | None:
    # 1) fenced code blocks: ```sql ... ``` or ``` ... ```
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) first occurrence starting with WITH/SELECT to the end
    m = re.search(r"(?:^|\n|\r)\s*(WITH|SELECT)\b[\s\S]*", text, flags=re.IGNORECASE)
    if m:
        return text[m.start():].strip()

    return None

# special verb we may add later to ask for clarification
if raw_text.upper().startswith("NEEDS_CLARIFICATION:"):
    if verbose:
        print("🛑 LLM requested clarification. Skipping execution.")
    return False, raw_text, "", None

sql_candidate = _extract_sql_from_text(raw_text)

if not sql_candidate:
    msg = ("The assistant did not return an executable SQL statement. "
           "Please ask a specific business question (metric, time window, breakdown).")
    if verbose:
        print("🛑 Could not extract SQL from LLM response. Skipping execution.")
    return False, msg, "", None

# Use the extracted SQL for execution
llm_response = sql_candidate
# --- END NEW ---

##======================= sql_agent.py==================

# --- NEW: Skip execution if the LLM didn't return SQL ---
sql_text = str(llm_response).strip()
import re
if sql_text.upper().startswith("NEEDS_CLARIFICATION:"):
    if verbose:
        print("🛑 LLM requested clarification. Skipping execution.")
    return False, sql_text, "", None

# naive but robust enough: SQL should start with WITH or SELECT
if not re.match(r'^\s*(WITH|SELECT)\b', sql_text, re.IGNORECASE):
    msg = ("The assistant could not produce SQL for this input. "
           "Please ask a specific business question (what metric, grain, time window).")
    if verbose:
        print("🛑 No SQL detected in LLM response. Skipping execution.")
    return False, msg, "", None
# --- END NEW ---


##============ Replace api chat fucntion ====================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"response": "Please enter a valid business question."}), 400

    # Lightweight intent guard for greetings / vague asks
    lq = question.lower()
    if lq in {"hi", "hello", "hey"} or lq.startswith(("hi ", "hello ", "hey ")):
        examples = [
            "Total demand sales and YoY by channel for Feb 2025",
            "Traffic and conversion rate by platform in FY26",
            "Trend sales by GMM for May 2025"
        ]
        return jsonify({
            "response": {
                "summary": "Hi! I analyze Sam’s e‑comm data. Try questions like:\n- " + "\n- ".join(examples),
                "dataframe": {"columns": [], "data": []}
            }
        }), 200

    try:
        success, result_df, log, sql, new_session_id = agent.process_user_query(message=question)

        if not success:
            # Always return a graceful message on failure
            return jsonify({
                "response": {
                    "summary": ("I couldn’t generate or run a query for that input. "
                                "Please be specific about metric, time window, and breakdown.\n"
                                "Examples:\n- Total demand sales and YoY by channel for Feb 2025\n"
                                "- Traffic and conversion rate by platform in FY26"),
                    "dataframe": {"columns": [], "data": []},
                    "agent_message": str(result_df),
                    "executed_sql": sql
                }
            }), 200

        # ---------- NEW: Insight Engine ----------
        facts, highlights = compute_insights(result_df, question)

        # Build a tighter system prompt for summarization
        summary_system = (
            "You are a senior e‑commerce analyst. Using ONLY the structured facts provided, "
            "write a concise executive summary for business stakeholders.\n"
            "- Bold key facts. Italicize insights.\n"
            "- Do not invent numbers or metrics. If a KPI is missing, omit it.\n"
            "- Keep to 8–12 short bullet lines, no tables.\n"
            "- End with 1–3 'Next actions' bullets if warranted."
        )

        # Pass pretty-printed facts (not raw DF) to SummaryAgent
        openai_summary_agent = SummaryAgent(
            model_type="openai",
            model_kwargs={"model": "gpt-4o", "temperature": 0.3}
        )

        facts_str = json.dumps({
            "question": question,
            "facts": facts
        }, indent=2)

        df_summary = openai_summary_agent.run(
            user_prompt="Produce the summary now using the provided facts.",
            input_data=facts_str,
            system_message=summary_system,
            verbose=True
        )

        # UI dataframe payload (stringify for safety)
        if isinstance(result_df, pd.DataFrame):
            result_df_json_friendly = result_df.astype(str)
            df_dict = result_df_json_friendly.to_dict(orient='split')
            df_payload = {"columns": df_dict['columns'], "data": df_dict['data']}
        else:
            df_payload = {"columns": [], "data": []}

        response_data = {
            "summary": df_summary,
            "highlights": highlights,   # <-- NEW for UI KPI cards
            "dataframe": df_payload
        }
        return jsonify({"response": response_data}), 200

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500


