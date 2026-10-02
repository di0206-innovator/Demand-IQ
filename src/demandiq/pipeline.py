"""End-to-end training, evaluation, and inventory intelligence pipeline for DemandIQ."""

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from demandiq.config import DEFAULT_CONFIG, AppConfig
from demandiq.data import load_raw_datasets, time_based_split
from demandiq.features import build_feature_pipeline
from demandiq.inventory import compute_inventory_summary_kpis, generate_inventory_recommendations_df
from demandiq.models import NaiveBaselineForecaster, XGBoostDemandForecaster
from demandiq.utils import set_seed, setup_logger

logger = setup_logger(__name__)


def generate_synthetic_sample_data(
    num_weeks: int = 145,
    num_centers: int = 5,
    num_meals: int = 8,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate deterministic synthetic food demand dataset matching Kaggle schemas."""
    set_seed(seed)
    np.random.seed(seed)

    # 1. Centers metadata
    center_ids = [10 + i for i in range(num_centers)]
    centers_df = pd.DataFrame(
        {
            "center_id": center_ids,
            "city_code": [500 + i for i in range(num_centers)],
            "region_code": [30 + (i % 3) for i in range(num_centers)],
            "center_type": ["TYPE_A" if i % 2 == 0 else "TYPE_B" for i in range(num_centers)],
            "op_area": [2.5 + 0.5 * i for i in range(num_centers)],
        }
    )

    # 2. Meals metadata
    meal_ids = [100 + i for i in range(num_meals)]
    categories = [
        "Beverages",
        "Rice Bowl",
        "Desert",
        "Pasta",
        "Sandwich",
        "Salad",
        "Pizza",
        "Seafood",
    ]
    cuisines = [
        "Indian",
        "Italian",
        "Continental",
        "Thai",
        "Indian",
        "Italian",
        "Continental",
        "Thai",
    ]
    meals_df = pd.DataFrame(
        {
            "meal_id": meal_ids,
            "category": categories[:num_meals],
            "cuisine": cuisines[:num_meals],
        }
    )

    # 3. Weekly transactions
    records: list[dict[str, Any]] = []
    record_id = 1000001

    base_prices = {m: 100.0 + (m % 10) * 25.0 for m in meal_ids}

    for week in range(1, num_weeks + 1):
        # Seasonality wave
        seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * (week % 52) / 52.0)
        for c in center_ids:
            for m in meal_ids:
                b_price = base_prices[m]
                discount = np.random.choice([0.0, 0.1, 0.2, 0.3], p=[0.5, 0.25, 0.15, 0.1])
                c_price = b_price * (1.0 - discount)

                email_promo = int(np.random.rand() < 0.15)
                home_promo = int(np.random.rand() < 0.10)

                # Demand generation logic with price elasticity and promo uplift
                base_demand = 80.0 + (m % 5) * 30.0 + (c % 3) * 20.0
                price_factor = np.exp(-1.5 * (c_price / b_price - 1.0))
                promo_factor = (1.4 if email_promo else 1.0) * (1.6 if home_promo else 1.0)
                noise = np.random.normal(1.0, 0.12)

                raw_orders = base_demand * seasonality * price_factor * promo_factor * noise
                orders = int(max(5, round(raw_orders)))

                records.append(
                    {
                        "id": record_id,
                        "week": week,
                        "center_id": c,
                        "meal_id": m,
                        "checkout_price": round(c_price, 2),
                        "base_price": round(b_price, 2),
                        "emailer_for_promotion": email_promo,
                        "homepage_featured": home_promo,
                        "num_orders": orders,
                    }
                )
                record_id += 1

    train_df = pd.DataFrame(records)
    logger.info(
        "Generated synthetic dataset: train=%d rows (%d weeks, %d centers, %d meals)",
        len(train_df),
        num_weeks,
        num_centers,
        num_meals,
    )
    return {
        "train": train_df,
        "centers": centers_df,
        "meals": meals_df,
    }


def run_training_pipeline(
    config: AppConfig = DEFAULT_CONFIG,
    use_synthetic_fallback: bool = True,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute complete demand forecasting and inventory intelligence pipeline."""
    set_seed(config.random_seed)
    logger.info("Starting DemandIQ End-to-End Pipeline...")

    # 1. Load Raw Datasets
    try:
        datasets = load_raw_datasets(config=config, include_test=False)
        logger.info("Loaded real datasets from %s", config.paths.data_raw)
    except FileNotFoundError as err:
        if use_synthetic_fallback:
            logger.warning(
                "Raw CSV files not found (%s). Initializing pipeline with synthetic dataset...",
                err,
            )
            datasets = generate_synthetic_sample_data(num_weeks=config.split.total_weeks)
        else:
            raise err

    # 2. Merge Transaction and Metadata Tables
    train_raw = datasets["train"]
    centers = datasets["centers"]
    meals = datasets["meals"]

    merged = train_raw.merge(centers, on="center_id", how="left")
    merged = merged.merge(meals, on="meal_id", how="left")

    # 3. Feature Engineering Pipeline
    engineered_df, feature_cols, mappings = build_feature_pipeline(
        raw_df=merged,
        config=config.feature,
    )

    # 4. Save Processed Dataset
    if save_artifacts:
        config.paths.data_processed.mkdir(parents=True, exist_ok=True)
        processed_path = config.paths.data_processed / "merged_features.parquet"
        engineered_df.to_parquet(processed_path, index=False)
        logger.info("Saved processed features to %s", processed_path)

    # 5. Time-Aware Train / Validation / Test Splitting
    train_split, val_split, test_split = time_based_split(
        engineered_df,
        time_col="week",
        config=config,
    )

    X_train = train_split[feature_cols]
    y_train = train_split[config.feature.target_col]

    X_val = val_split[feature_cols]
    y_val = val_split[config.feature.target_col]

    X_test = test_split[feature_cols]
    y_test = test_split[config.feature.target_col]

    # 6. Model 1: Naive Baseline Forecaster
    logger.info("Training Naive Baseline Forecaster...")
    baseline = NaiveBaselineForecaster(group_cols=config.feature.group_cols)
    baseline.fit(X_train, y_train)
    baseline_metrics = baseline.evaluate(X_test, y_test)
    logger.info("Baseline Test Metrics: %s", baseline_metrics)

    # 7. Model 2: XGBoost Demand Forecaster
    logger.info("Training XGBoost Demand Forecaster...")
    xgb_model = XGBoostDemandForecaster(
        model_config=config.model,
        use_log_target=config.feature.use_log_target,
    )
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    logger.info("XGBoost Test Metrics: %s", xgb_metrics)

    # 8. Feature Importance & SHAP
    top_importances = xgb_model.get_feature_importances(top_n=15)
    logger.info("Top Features: %s", list(top_importances.keys())[:5])

    # 9. Inventory Recommendation Generation on Test Split
    test_forecasts = xgb_model.predict(X_test)
    test_eval_df = test_split.copy()
    test_eval_df["forecast_orders"] = test_forecasts
    test_eval_df["actual_orders"] = y_test.to_numpy()

    # Calculate model residual std on validation split for safety stock buffer
    val_preds = xgb_model.predict(X_val)
    val_residuals = y_val.to_numpy() - val_preds
    model_rmse = float(np.sqrt(np.mean(val_residuals**2)))

    recommendations_df = generate_inventory_recommendations_df(
        test_eval_df,
        forecast_col="forecast_orders",
        demand_std_default=model_rmse,
        config=config.inventory,
    )
    inventory_kpis = compute_inventory_summary_kpis(recommendations_df)
    logger.info("Inventory KPIs: %s", inventory_kpis)

    # 10. Persist Artifacts
    if save_artifacts:
        config.paths.models.mkdir(parents=True, exist_ok=True)
        model_file = config.paths.models / "xgboost_model.joblib"
        xgb_model.save(model_file)

        metrics_payload = {
            "baseline_metrics": baseline_metrics,
            "xgboost_metrics": xgb_metrics,
            "inventory_kpis": inventory_kpis,
            "top_features": top_importances,
            "category_mappings": {
                col: {str(k): int(v) for k, v in m.items()} for col, m in mappings.items()
            },
        }
        metrics_file = config.paths.models / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        recs_file = config.paths.data_processed / "inventory_recommendations.csv"
        recommendations_df.to_csv(recs_file, index=False)
        logger.info("Persisted pipeline summary and recommendations to %s", recs_file)

    return {
        "baseline_metrics": baseline_metrics,
        "xgboost_metrics": xgb_metrics,
        "inventory_kpis": inventory_kpis,
        "top_features": top_importances,
        "feature_count": len(feature_cols),
        "train_rows": len(train_split),
        "test_rows": len(test_split),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DemandIQ training and inventory pipeline")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force running pipeline on synthetic benchmark data",
    )
    args = parser.parse_args()

    results = run_training_pipeline(use_synthetic_fallback=args.synthetic or True)
    print("\n" + "=" * 60)
    print("📈 DEMANDIQ PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Features Extracted : {results['feature_count']}")
    print(f"Training Samples   : {results['train_rows']}")
    print(f"Test Samples       : {results['test_rows']}")
    print("-" * 60)
    print("MODEL COMPARISON (Test Set):")
    print(
        f"  Naive Baseline   -> WMAPE: {results['baseline_metrics']['WMAPE']} | RMSLE: {results['baseline_metrics']['RMSLE']} | MAE: {results['baseline_metrics']['MAE']}"
    )
    print(
        f"  XGBoost Forecast -> WMAPE: {results['xgboost_metrics']['WMAPE']} | RMSLE: {results['xgboost_metrics']['RMSLE']} | MAE: {results['xgboost_metrics']['MAE']}"
    )
    print("-" * 60)
    print("INVENTORY RECOMMENDATIONS:")
    for k, v in results["inventory_kpis"].items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
