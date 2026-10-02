"""Tests for end-to-end pipeline execution and synthetic data generation."""

from pathlib import Path

from demandiq.config import AppConfig, ModelConfig, PathConfig, SplitConfig
from demandiq.pipeline import generate_synthetic_sample_data, run_training_pipeline


def test_generate_synthetic_sample_data():
    datasets = generate_synthetic_sample_data(num_weeks=20, num_centers=2, num_meals=3)
    assert set(datasets.keys()) == {"train", "centers", "meals"}
    assert len(datasets["centers"]) == 2
    assert len(datasets["meals"]) == 3
    assert len(datasets["train"]) == 20 * 2 * 3  # 120 rows
    assert "num_orders" in datasets["train"].columns
    assert (datasets["train"]["num_orders"] > 0).all()


def test_run_training_pipeline_end_to_end(tmp_path: Path):
    paths = PathConfig()
    object.__setattr__(paths, "project_root", tmp_path)
    object.__setattr__(paths, "data_raw", tmp_path / "data" / "raw")
    object.__setattr__(paths, "data_processed", tmp_path / "data" / "processed")
    object.__setattr__(paths, "models", tmp_path / "models")
    object.__setattr__(paths, "reports_figures", tmp_path / "reports" / "figures")

    split = SplitConfig(train_end_week=15, val_end_week=20, total_weeks=25)
    model_cfg = ModelConfig(
        n_estimators=20, max_depth=3, learning_rate=0.1, early_stopping_rounds=5
    )
    config = AppConfig(paths=paths, split=split, model=model_cfg)

    results = run_training_pipeline(
        config=config,
        use_synthetic_fallback=True,
        save_artifacts=True,
    )

    # Check execution results
    assert "baseline_metrics" in results
    assert "xgboost_metrics" in results
    assert "inventory_kpis" in results
    assert results["feature_count"] > 5
    assert results["train_rows"] > 0
    assert results["test_rows"] > 0

    # Verify artifacts were written to disk
    assert (paths.data_processed / "merged_features.parquet").exists()
    assert (paths.models / "xgboost_model.joblib").exists()
    assert (paths.models / "metrics.json").exists()
    assert (paths.data_processed / "inventory_recommendations.csv").exists()
