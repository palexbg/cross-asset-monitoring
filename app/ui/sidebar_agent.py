import streamlit as st
from backend.ai.agent import PortfolioAgent
from backend.perfstats import PortfolioStats
from backend.config import FactorRiskConfig


def render_ai_analyst_sidebar(context=None):
    if context is None:
        context = {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Ask the AI Analyst")

    # Check for API Key
    api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input(
        "OpenAI API Key", type="password")

    if not api_key:
        st.sidebar.warning("Enter API Key to chat.")
        return

    # Initialize AI Analyst
    if "ai_analyst" not in st.session_state:
        try:
            st.session_state["ai_analyst"] = PortfolioAgent(
                api_key=api_key, context=context)
            st.session_state["chat_history"] = []
            st.session_state["ai_analyst_error"] = None
        except Exception as e:
            st.session_state["ai_analyst_error"] = str(e)

    # Every re-run updates the context
    if "ai_analyst" in st.session_state:
        st.session_state["ai_analyst"].update_context(context)

    if st.session_state.get("ai_analyst_error"):
        st.sidebar.error(f"Error: {st.session_state['ai_analyst_error']}")
        return

    # The 'height' parameter makes it scroll internally.
    chat_container = st.sidebar.container(height=200, border=True)

    # render History inside the container
    with chat_container:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # chat input
    if prompt := st.sidebar.chat_input("Ask about risk..."):
        # append user message to state immediately so it persists
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt})

        # render the new user message immediately in the container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # render Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state["ai_analyst"].ask(prompt)
                        st.markdown(response)

                        # Save assistant response to history
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"Error: {e}")


def prepare_context_agent(pf_analytics: dict,
                          base_currency: str,
                          portfolio_preset: str,
                          start_date: str,
                          end_date: str) -> dict:
    """
    Prepare the context dictionary for the AI Analyst sidebar.
    """
    agent_context = {
        "Base Currency": base_currency,
        "Portfolio Name": portfolio_preset,
        "Analysis Period": f"From {str(start_date)} to {str(end_date)}",
        "summary_stats": {}
    }

    if pf_analytics:
        # Summary Stats
        bt = pf_analytics["bt"]
        rf = pf_analytics["rf"]
        port_stats = PortfolioStats(backtest_result=bt, risk_free=rf)
        summary_df = port_stats.summary_table()

        if not summary_df.empty:
            stats_dict = dict(zip(summary_df["Metric"], summary_df["Value"]))

            # Add Current Drawdown
            nav = bt.nav
            peak = nav.expanding().max()
            curr_dd = (nav.iloc[-1] / peak.iloc[-1]) - 1.0
            stats_dict["Current Drawdown"] = f"{curr_dd:.2%}"

            agent_context["summary_stats"] = stats_dict

        # Current Holdings (Full Table)
        last_w = bt.weights.iloc[-1]
        # clean up zero weights
        full_weights = last_w[last_w != 0].dropna()

        agent_context[f"Current Weights (as of {last_w.name.date()})"] = \
            full_weights.apply(lambda x: f"{x:.4%}").to_dict()

        # the two risk profiles - ractor risk and asset (holdings) risk
        fr = pf_analytics.get("factor_risk")
        ar = pf_analytics.get("asset_risk")

        if fr and ar:
            # var decomposition
            sys_var = fr.get("latest_systematic_vol", 0.0) ** 2
            idio_var = fr.get("latest_idio_vol", 0.0) ** 2
            model_total_var = sys_var + idio_var
            actual_total_vol = ar.get("latest_port_vol", 0.0)

            if model_total_var > 0:
                sys_pct = sys_var / model_total_var
                idio_pct = idio_var / model_total_var
            else:
                sys_pct, idio_pct = 0.0, 0.0

            risk_summary = {
                "Total Realized Volatility": f"{actual_total_vol:.2%}",
                f"Risk Decomposition (Variance Explained)": {
                    "Systematic (Market)": f"{sys_pct:.1%}",
                    "Idiosyncratic (Specific)": f"{idio_pct:.1%}"
                }
            }

            # factor risk table, columns are renamed for the llm to parse them more easily
            if 'latest_factor_rc' in fr:
                factor_df = fr['latest_factor_rc'].copy()
                factor_df.sort_values("ctr_pct", ascending=False, inplace=True)

                # explicit renaming to prevent "beta vs risk" confusion
                factor_df.columns = [
                    "Factor Beta",
                    "MCTR (Marginal)",
                    "CTR (Absolute)",
                    "Risk Contrib %"
                ]

                # to markdown
                risk_summary["Systematic Risk Detail (Table)"] = \
                    factor_df.to_markdown(floatfmt=".4f")

            # asset risk table, columns are renamed for the llm to parse them more easily
            if 'latest_rc' in ar:
                asset_df = ar['latest_rc'].copy()
                asset_df.sort_values("ctr_pct", ascending=False, inplace=True)

                asset_df.columns = [
                    "Weight",
                    "MCTR (Marginal)",
                    "CTR (Absolute)",
                    "Risk Contrib %"
                ]

                risk_summary["Asset Risk Detail (Table)"] = \
                    asset_df.to_markdown(floatfmt=".4f")

            agent_context["Risk Profile"] = risk_summary

        # factor exposures (beta + t-stat combined)
        if 'betas' in pf_analytics and 't_stats' in pf_analytics:
            last_beta = pf_analytics['betas'].iloc[-1]
            last_t = pf_analytics['t_stats'].iloc[-1]

            # Combine into a clean dictionary
            full_exposures = {}
            for factor, beta in last_beta.items():
                t_stat = last_t.get(factor, 0)
                sig = "(Sig)" if abs(t_stat) > 1.96 else ""
                full_exposures[factor] = f"{beta:.2f} (t={t_stat:.1f}) {sig}"

            agent_context["Factor Exposures"] = full_exposures

    return agent_context
