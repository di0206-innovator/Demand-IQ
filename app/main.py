"""DemandIQ Streamlit Application entrypoint placeholder.

Full dashboard implementation is scheduled for Phase 4.
"""

import streamlit as st

from demandiq import __version__


def main() -> None:
    st.set_page_config(
        page_title="DemandIQ",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 DemandIQ: AI Demand Forecasting & Inventory Intelligence")
    st.caption(f"DemandIQ v{__version__} | Status: Phase 0 Scaffolding Complete")
    st.info(
        "Application placeholder initialized. The complete interactive dashboard with "
        "forecasting, explainability, and inventory recommendations will be activated in Phase 4."
    )


if __name__ == "__main__":
    main()
