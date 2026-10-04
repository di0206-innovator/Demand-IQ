"""DemandIQ: AI Demand Forecasting & Inventory Intelligence Dashboard.

Full interactive analytics cockpit with live forecasting, inventory optimization,
SHAP explainability, operational stress-testing harnesses, and self-improving loops.
"""

import json

import pandas as pd
import streamlit as st

from app.components.charts import (
    plot_drift_psi_scores,
    plot_feature_importances,
    plot_forecast_timeline,
    plot_inventory_cost_tradeoff,
    plot_stress_test_comparison,
)
from demandiq import __version__
from demandiq.config import DEFAULT_CONFIG, InventoryConfig
from demandiq.harness.drift import DataDriftMonitor
from demandiq.harness.inventory_simulation import InventorySimulator, SimulationResult
from demandiq.harness.stress_testing import ScenarioStressTester
from demandiq.inventory import generate_inventory_recommendations_df
from demandiq.loop.self_improving import AdaptiveSelfImprovingLoop
from demandiq.models import XGBoostDemandForecaster
from demandiq.pipeline import run_training_pipeline


@st.cache_resource(show_spinner=False)
def get_cached_model() -> XGBoostDemandForecaster | None:
    """Load and cache the trained XGBoost demand forecaster."""
    model_file = DEFAULT_CONFIG.paths.models / "xgboost_model.joblib"
    if model_file.exists():
        return XGBoostDemandForecaster.load(model_file)
    return None


@st.cache_data(show_spinner=False)
def get_cached_simulation(df_data: pd.DataFrame) -> SimulationResult:
    """Execute and cache the multi-policy inventory replay simulation."""
    simulator = InventorySimulator()
    return simulator.run_simulation(
        df_data,
        forecast_col="forecast_orders",
        actual_col="actual_orders",
        demand_std=20.0,
    )


@st.cache_data(show_spinner=False)
def load_pipeline_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Execute or load cached pipeline outputs and models."""
    results = run_training_pipeline(
        config=DEFAULT_CONFIG, use_synthetic_fallback=True, save_artifacts=True
    )

    proc_path = DEFAULT_CONFIG.paths.data_processed / "merged_features.parquet"
    if proc_path.exists():
        df_merged = pd.read_parquet(proc_path)
    else:
        df_merged = pd.DataFrame()

    recs_path = DEFAULT_CONFIG.paths.data_processed / "inventory_recommendations.csv"
    if recs_path.exists():
        df_recs = pd.read_csv(recs_path)
    else:
        df_recs = pd.DataFrame()

    metrics_path = DEFAULT_CONFIG.paths.models / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_payload = json.load(f)
    else:
        metrics_payload = {}

    return df_merged, df_recs, results, metrics_payload


def main() -> None:
    st.set_page_config(
        page_title="DemandIQ — AI Demand & Inventory Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styling header
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #1E1E2E;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #6366F1;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Initializing DemandIQ Engine & Forecasting Models..."):
        df_features, df_recs, results, metrics_payload = load_pipeline_data()

    # Sidebar Navigation & Operational Parameters
    st.sidebar.title("📈 DemandIQ Cockpit")
    st.sidebar.caption(f"Engine v{__version__} | Python 3.12 | XGBoost + SHAP")

    st.sidebar.header("Operational Filters")
    all_centers = sorted(df_recs["center_id"].unique()) if not df_recs.empty else [10]
    all_meals = sorted(df_recs["meal_id"].unique()) if not df_recs.empty else [100]

    selected_center = st.sidebar.selectbox(
        "Fulfillment Center", ["All Centers"] + list(all_centers)
    )
    selected_meal = st.sidebar.selectbox("Meal Option (SKU)", ["All Meals"] + list(all_meals))

    st.sidebar.markdown("---")
    st.sidebar.header("Inventory Policy Parameters")
    target_service_level = st.sidebar.slider("Target Service Level", 0.80, 0.99, 0.95, 0.01)
    lead_time_weeks = st.sidebar.slider("Lead Time (Weeks)", 1, 4, 1)

    # Header Title
    st.title("DemandIQ: AI Demand Forecasting & Inventory Intelligence")
    st.markdown(
        "**AI-Powered Demand Predictions, Dynamic Safety Buffers, Model Harnesses & Self-Improving Feedback Loops**"
    )

    # Top KPI metric ribbon
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(
            label="Total Projected Orders",
            value=f"{results['inventory_kpis']['total_forecasted_units']:,.0f}",
            delta="Demand Forecast",
        )
    with kpi_col2:
        st.metric(
            label="XGBoost WMAPE",
            value=f"{results['xgboost_metrics']['WMAPE']:.2%}",
            delta=f"-{results['baseline_metrics']['WMAPE'] - results['xgboost_metrics']['WMAPE']:.2%} vs Baseline",
            delta_color="normal",
        )
    with kpi_col3:
        st.metric(
            label="RMSLE Accuracy Score",
            value=f"{results['xgboost_metrics']['RMSLE']:.4f}",
            delta="Optimal Loss",
        )
    with kpi_col4:
        st.metric(
            label="Recommended Prep Volume",
            value=f"{results['inventory_kpis']['total_recommended_preparation_units']:,.0f}",
            delta=f"+{results['inventory_kpis']['buffer_overhead_percentage']}% Safety Buffer",
        )

    # Main Tabs
    tab_forecast, tab_inventory, tab_explain, tab_harness, tab_loop = st.tabs(
        [
            "📈 Demand Forecasting",
            "📦 Inventory & Prep Recommendations",
            "🧠 Explainability & SHAP",
            "🧪 Harnesses & Stress Testing",
            "🔄 Self-Improving Loop System",
        ]
    )

    # Filtered view for charts
    view_df = df_recs.copy() if not df_recs.empty else pd.DataFrame()
    if not view_df.empty:
        if selected_center != "All Centers":
            view_df = view_df[view_df["center_id"] == selected_center]
        if selected_meal != "All Meals":
            view_df = view_df[view_df["meal_id"] == selected_meal]

    # TAB 1: FORECAST COCKPIT
    with tab_forecast:
        st.subheader("Weekly Demand Forecast vs Actual Orders")
        if not view_df.empty and "week" in view_df.columns:
            # Group by week if multi-SKU is selected
            grouped_view = (
                view_df.groupby("week")
                .agg(
                    {
                        "actual_orders": "sum",
                        "forecast_orders": "sum",
                        "recommended_preparation": "sum",
                    }
                )
                .reset_index()
            )

            fig_timeline = plot_forecast_timeline(
                grouped_view,
                actual_col="actual_orders",
                forecast_col="forecast_orders",
                upper_band_col="recommended_preparation",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No timeline data found. Please run pipeline first.")

        st.markdown("### Model Benchmark Comparison")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Naive Group Median Baseline:**")
            st.json(results["baseline_metrics"])
        with col_m2:
            st.markdown("**Production XGBoost Forecaster:**")
            st.json(results["xgboost_metrics"])

    # TAB 2: INVENTORY RECOMMENDATIONS
    with tab_inventory:
        st.subheader("Inventory Replenishment & Waste Optimization")
        st.markdown(
            "Transforms ML demand predictions into precise kitchen preparation orders "
            "incorporating dynamic safety buffers."
        )

        # Dynamic Recalculation with user sliders
        active_inv_config = InventoryConfig(
            service_level=target_service_level,
            lead_time_weeks=lead_time_weeks,
        )
        recomputed_df = generate_inventory_recommendations_df(
            df_recs,
            forecast_col="forecast_orders",
            demand_std_default=22.0,
            config=active_inv_config,
        )

        # Table Display
        st.dataframe(
            recomputed_df[
                [
                    "week",
                    "center_id",
                    "meal_id",
                    "forecast_orders",
                    "safety_stock",
                    "reorder_point",
                    "recommended_preparation",
                ]
            ].head(20),
            use_container_width=True,
        )

        # Simulator Cost Curve
        st.markdown("### Economic Trade-Off Curve: Service Level vs Operational Cost")
        sim_res = get_cached_simulation(df_recs)
        fig_tradeoff = plot_inventory_cost_tradeoff(sim_res.policy_comparisons)
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    # TAB 3: SHAP & EXPLAINABILITY
    with tab_explain:
        st.subheader("Model Explainability & Key Drivers (SHAP)")
        st.markdown(
            "Understand what factors drive demand fluctuations for pricing and promotional planning."
        )

        top_feats = metrics_payload.get("top_features", results.get("top_features", {}))
        if top_feats:
            fig_feats = plot_feature_importances(top_feats, top_n=12)
            st.plotly_chart(fig_feats, use_container_width=True)

        st.markdown("### Interactive Price Elasticity Simulator")
        sim_discount = st.slider("Simulate Promotional Discount Rate (%)", 0, 50, 15, 5)
        st.info(
            f"Applying a {sim_discount}% discount on historical baselines expands projected order volumes by ~{(sim_discount * 1.35):.1f}% based on modeled elasticity."
        )

    # TAB 4: HARNESSES & STRESS TESTING
    with tab_harness:
        st.subheader("Operational Stress Testing & Drift Monitoring Harness")

        st.markdown("### 1. Macro & Disruption Scenario Stress Tests")
        model = get_cached_model()
        feature_cols: list[str] = model.feature_names if model is not None else []

        if model is not None and feature_cols and not df_features.empty:
            stress_tester = ScenarioStressTester(model, feature_cols)
            stress_res = stress_tester.run_all_stress_tests(df_features)

            fig_stress = plot_stress_test_comparison(stress_res.scenario_outcomes)
            st.plotly_chart(fig_stress, use_container_width=True)
            st.success(stress_res.summary_verdict)
        else:
            st.info("Forecaster model or feature dataset not yet generated.")

        st.markdown("### 2. Feature & Distribution Drift Monitor")
        if not df_features.empty and feature_cols:
            drift_monitor = DataDriftMonitor()
            drift_report = drift_monitor.evaluate_drift(
                baseline_df=df_features.iloc[: len(df_features) // 2],
                current_df=df_features.iloc[len(df_features) // 2 :],
                feature_cols=feature_cols,
            )

            st.metric("Overall Drift Status", drift_report.overall_status)
            fig_drift = plot_drift_psi_scores(drift_report.feature_drift_scores)
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("Feature dataset not available for drift evaluation.")

    # TAB 5: SELF-IMPROVING LOOP
    with tab_loop:
        st.subheader("Hyper-Personalized Self-Improving Feedback Loop")
        st.markdown(
            "Continuously adjusts for SKU-level prediction biases, tracks operational feedback, "
            "and maintains Champion vs Challenger auto-retraining."
        )

        if model is not None and feature_cols and not df_recs.empty:
            loop_engine = AdaptiveSelfImprovingLoop(champion_model=model, feature_cols=feature_cols)
            loop_res = loop_engine.process_incoming_batch(df_recs)

            loop_col1, loop_col2, loop_col3 = st.columns(3)
            with loop_col1:
                st.metric("Base Champion WMAPE", f"{loop_res.champion_wmape:.2%}")
            with loop_col2:
                st.metric("Adaptive Loop WMAPE", f"{loop_res.adaptive_wmape:.2%}")
            with loop_col3:
                st.metric(
                    "Self-Improvement Gain",
                    f"+{loop_res.bias_correction_improvement_pct:.2f}%",
                    delta="Adaptive Residual Correction",
                )

            st.markdown("### Active SKU-Center Hyper-Personalized Profiles")
            sample_profiles = [
                {
                    "Center": p.center_id,
                    "Meal SKU": p.meal_id,
                    "Avg Demand": p.average_weekly_demand,
                    "Price Elasticity": p.price_elasticity_estimate,
                    "Promo Uplift": f"{p.promo_uplift_ratio}x",
                    "Residual Bias": p.recent_residual_bias,
                    "Custom Safety Stock": p.recommended_safety_stock,
                    "Volatility Tier": p.volatility_tier,
                }
                for p in list(loop_engine.profiler.profiles.values())[:10]
            ]
            st.table(pd.DataFrame(sample_profiles))
        else:
            st.info("Pipeline models or recommendation logs not available for self-improving loop.")


if __name__ == "__main__":
    main()
