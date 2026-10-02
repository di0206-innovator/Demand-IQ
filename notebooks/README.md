# DemandIQ Notebooks

This directory contains research, exploratory data analysis (EDA), and experimental notebooks.

## Planned Notebooks
- `01_exploratory_data_analysis.ipynb`: Comprehensive exploratory data analysis planned for execution in Google Colab / local Jupyter.
  - Distribution of demand (`num_orders`), skewness, log-transform analysis.
  - Granularity analysis (fulfillment center types, cities, meal categories, cuisine demand).
  - Price elasticity analysis (`checkout_price` vs `base_price`, promotional uplift).
  - Temporal patterns, seasonality, and stationarity checks.
- `02_model_experimentation.ipynb`: Prototyping baseline vs. XGBoost model, feature importance, and SHAP explainability.

## Rules
- Production code must not reside solely in notebooks; reusable logic should be implemented in `src/demandiq/` and imported into notebooks.
- Notebook outputs containing sensitive data should be cleared before committing.
