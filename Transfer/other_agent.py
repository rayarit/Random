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
