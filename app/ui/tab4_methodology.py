import streamlit as st


def render_methodology_tab():
    """Render the methodology / info tab.

    This is a read-only view that loads the Markdown explanation of the
    factor lens methodology from the `docs` folder so it can be maintained
    separately from the UI code.
    """

    st.title("Methodology – Factor Lens")
    # Top disclaimer (clearly visible)
    st.markdown("---")
    st.info(
        "**Disclaimer:** For educational & demonstrational purposes only. "
        "Loads a local CSV of public prices sourced from Yahoo Finance; not investment advice."
    )

    try:
        with open("docs/factor_methodology.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        st.error(
            "Could not find `docs/factor_methodology.md`. Please ensure the file exists.")
        return

    st.markdown(content)
