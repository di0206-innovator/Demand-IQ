"""Interactive visualization charts using Plotly for DemandIQ dashboard."""

import pandas as pd
import plotly.graph_objects as go


def plot_forecast_timeline(
    df: pd.DataFrame,
    time_col: str = "week",
    actual_col: str = "actual_orders",
    forecast_col: str = "forecast_orders",
    upper_band_col: str | None = "recommended_preparation",
    title: str = "Weekly Demand Forecast & Kitchen Preparation Target",
) -> go.Figure:
    """Create interactive time series chart comparing actuals, forecast, and preparation upper buffer."""
    fig = go.Figure()

    # Upper preparation buffer area
    if upper_band_col and upper_band_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[upper_band_col],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                name="Prep Target Upper",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[forecast_col],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(99, 102, 241, 0.15)",
                line={"color": "rgba(99, 102, 241, 0.8)", "width": 2},
                name="Safety Buffer / Prep Target",
            )
        )

    # Actual historical orders
    if actual_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[actual_col],
                mode="lines+markers",
                marker={"size": 6, "color": "#10B981"},
                line={"color": "#10B981", "width": 2.5},
                name="Actual Demand (Orders)",
            )
        )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[forecast_col],
            mode="lines+markers",
            marker={"size": 5, "color": "#6366F1"},
            line={"color": "#6366F1", "width": 2.5, "dash": "dash"},
            name="XGBoost ML Forecast",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Timeline (Week)",
        yaxis_title="Demand (Order Units)",
        template="plotly_dark",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_feature_importances(importances: dict[str, float], top_n: int = 12) -> go.Figure:
    """Create horizontal bar chart of top feature drivers."""
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in reversed(items)]
    scores = [item[1] for item in reversed(items)]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker={
                "color": scores,
                "colorscale": "Viridis",
                "showscale": False,
            },
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Demand Drivers (Feature Gain)",
        xaxis_title="Relative Gain / Importance",
        yaxis_title="Feature Name",
        template="plotly_dark",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


def plot_inventory_cost_tradeoff(policy_comparisons: list) -> go.Figure:
    """Plot total operational cost curve vs service level targets."""
    sl_vals = [p.service_level * 100 for p in policy_comparisons]
    holding_costs = [p.estimated_holding_cost for p in policy_comparisons]
    stockout_costs = [p.estimated_stockout_loss for p in policy_comparisons]
    total_costs = [p.total_operational_cost for p in policy_comparisons]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sl_vals,
            y=holding_costs,
            mode="lines+markers",
            name="Holding / Spoilage Cost ($)",
            line={"color": "#F59E0B", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sl_vals,
            y=stockout_costs,
            mode="lines+markers",
            name="Stockout Penalty / Lost Sales ($)",
            line={"color": "#EF4444", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sl_vals,
            y=total_costs,
            mode="lines+markers",
            name="Total Operational Cost ($)",
            line={"color": "#6366F1", "width": 3, "dash": "solid"},
        )
    )

    fig.update_layout(
        title="Inventory Cost Optimization Curve: Service Level vs Economic Waste",
        xaxis_title="Target Cycle Service Level (%)",
        yaxis_title="Estimated Weekly Cost ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_stress_test_comparison(outcomes: list) -> go.Figure:
    """Plot scenario disruption comparison against baseline."""
    names = [o.scenario_name for o in outcomes]
    base_demands = [o.baseline_total_demand for o in outcomes]
    stressed_demands = [o.stressed_total_demand for o in outcomes]

    fig = go.Figure(
        data=[
            go.Bar(name="Baseline Demand", x=names, y=base_demands, marker_color="#4B5563"),
            go.Bar(
                name="Stressed Scenario Demand", x=names, y=stressed_demands, marker_color="#8B5CF6"
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Operational Disruption Scenarios (Demand Impact)",
        xaxis_title="Shock Scenario",
        yaxis_title="Total Projected Orders",
        template="plotly_dark",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_drift_psi_scores(drift_scores: list) -> go.Figure:
    """Plot feature Population Stability Index (PSI) scores with threshold lines."""
    names = [d.feature_name for d in drift_scores[:15]]
    psi_vals = [d.psi_score for d in drift_scores[:15]]
    colors = ["#EF4444" if p >= 0.20 else ("#F59E0B" if p >= 0.10 else "#10B981") for p in psi_vals]

    fig = go.Figure(go.Bar(x=names, y=psi_vals, marker_color=colors))
    fig.add_hline(
        y=0.20,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Critical Drift (PSI >= 0.20)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.10,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text="Moderate Drift (PSI >= 0.10)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Feature Population Stability Index (PSI) Drift Monitor",
        xaxis_title="Engineered Feature",
        yaxis_title="PSI Score",
        template="plotly_dark",
        margin={"l": 40, "r": 40, "t": 50, "b": 80},
    )
    return fig
