# DemandIQ: AI Demand Forecasting & Inventory Intelligence

[![CI](https://github.com/di0206-innovator/Demand-IQ/actions/workflows/ci.yml/badge.svg)](https://github.com/di0206-innovator/Demand-IQ/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: Pytest](https://img.shields.io/badge/tested%20with-pytest-blue.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, production-grade machine learning system that transforms historical food fulfillment demand data into high-accuracy forecasts and automated kitchen inventory/preparation recommendations.

---

## 🎯 Executive Summary

In meal preparation and fulfillment centers, forecasting errors directly produce two costly operational failure modes:
1. **Under-forecasting:** Food stockouts, unmet customer demand, and lost revenue.
2. **Over-forecasting:** Food spoilage, excessive holding costs, and operational waste.

**DemandIQ** bridges the gap between predictive ML and operational supply chain decision-making. Rather than stopping at raw point forecasts, DemandIQ couples an optimized **XGBoost** demand model with a statistical **Safety Stock & Inventory Recommendation Engine**, offering fulfillment managers actionable preparation guidance with uncertainty buffers.

---

## 🏛️ System Architecture

```text
demandiq/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI (Ruff, Pytest, uv)
├── app/
│   ├── __init__.py
│   ├── main.py                  # Streamlit Dashboard application
│   └── README.md
├── data/
│   ├── raw/                     # Immutable raw datasets (train.csv, centers, meals)
│   │   ├── .gitkeep
│   │   └── README.md            # Schema and Kaggle download instructions
│   └── processed/               # Transformed parquet/csv datasets & splits
│       ├── .gitkeep
│       └── README.md
├── notebooks/                   # Research, Colab EDA, and prototyping notebooks
│   ├── .gitkeep
│   └── README.md
├── reports/
│   └── figures/                 # EDA plots, feature importances, evaluation charts
│       └── .gitkeep
├── models/                      # Trained model weights & serialized pipelines
│   └── .gitkeep
├── src/
│   └── demandiq/
│       ├── __init__.py          # Package initialization & versioning
│       ├── config.py            # Central dataclass configs (paths, splits, inventory)
│       ├── data.py              # Data loaders, schema validation, temporal splitters
│       ├── features.py          # Leak-free feature transformers & lag generators
│       ├── models.py            # Baseline forecaster and XGBoost model interfaces
│       ├── inventory.py         # Statistical safety stock & order recommendations
│       └── utils.py             # Metrics (WMAPE, RMSLE), logging, and reproducibility
├── tests/                       # Unit and integration test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_inventory.py
│   ├── test_models.py
│   └── test_utils.py
├── .gitignore                   # Rigorous gitignore for data, models, and caches
├── .python-version              # Python 3.12 pin
├── pyproject.toml               # Modern PEP 621 packaging & tool configs
├── LICENSE                      # MIT Open Source License
└── README.md
```

---

## 🛠️ Technology Stack

| Domain | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Modern syntax, improved performance, strict type annotations. |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) | Extremely fast, deterministic, cross-platform dependency resolution. |
| **Data Processing** | Pandas, NumPy | High-performance tabular transformation and vector operations. |
| **Machine Learning** | scikit-learn, XGBoost | Gradient boosted trees for tabular time-series demand forecasting. |
| **Model Explainability** | SHAP | Shapley additive explanations for driver attribution and transparency. |
| **Inventory Optimization** | Custom Statistical Engine | Normal inverse CDF service level calculations for dynamic safety stock. |
| **Interactive UI** | Streamlit, Plotly | Interactive dashboard showcasing forecast, intervals, and recommendations. |
| **Quality & Testing** | Ruff, Pytest, Pytest-Cov | Fast linting, formatting, and unit test verification. |
| **CI/CD** | GitHub Actions | Automated quality gates on every commit and pull request. |

---

## 🔬 Core Engineering Principles

1. **Strict Temporal Splitting:** Time-aware train/val/test splits (Weeks 1–125 train, 126–135 val, 136–145 test) to prevent future-data leakage.
2. **Lag Safety:** All lag and rolling features enforce a minimum lag of $\ge 1$ period.
3. **Dual Metric Evaluation:** Evaluated using business-relevant metrics:
   - **WMAPE** ($\sum |y - \hat{y}| / \sum |y|$): Handles varied volume scales without percentage bias.
   - **RMSLE** ($\sqrt{\frac{1}{n} \sum (\log(1+\hat{y}) - \log(1+y))^2}$): Penalizes relative error and protects against large outlier skew.
   - **MAE & RMSE**: Standard absolute error tracking.
4. **Actionable Decision Layer:** Forecasts translate into net preparation units based on cycle service level targets (e.g. 95%) and safety stock buffer:
   $$\text{Safety Stock} = Z \times \sqrt{L} \times \sigma_D$$
   $$\text{Recommended Preparation} = \max(0, \text{Forecast} + \text{Safety Stock} - \text{Current Inventory})$$

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) (version 0.4+ recommended)
- Python 3.12

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/di0206-innovator/Demand-IQ.git
cd Demand-IQ

# Create virtual environment and synchronize dependencies via uv
uv sync --extra dev
```

### 2. Activate Virtual Environment (Optional)
```bash
source .venv/bin/activate
```
*(Or prepend `uv run` to any command).*

### 3. Data Download
Refer to [`data/raw/README.md`](data/raw/README.md) for downloading the Kaggle Food Demand Challenge CSV files into `data/raw/`:
- `train.csv`
- `fulfilment_center_info.csv`
- `meal_info.csv`

---

## 🧪 Quality Assurance & Testing

Run code quality and test commands:

```bash
# Run Ruff linting
uv run ruff check .

# Run Ruff formatting verification
uv run ruff format --check .

# Run Pytest suite with coverage report
uv run pytest --cov=demandiq --cov-report=term-missing
```

---

## 🗺️ Project Execution Roadmap

- [x] **Phase 0: Architecture & Environment Scaffolding**
  - Project directory hierarchy, packaging configuration (`pyproject.toml`), `.gitignore`, data documentation, CI pipeline, type-annotated core modules, and comprehensive unit tests.
- [ ] **Phase 1: Exploratory Data Analysis (EDA)**
  - Granularity profiling, price elasticity analysis, promo uplift, demand distribution transformations, and stationarity diagnostics.
- [ ] **Phase 2: Feature Engineering & Forecasting Model**
  - Baseline comparison, temporal cross-validation, hyperparameter-tuned XGBoost model, and SHAP explainability.
- [ ] **Phase 3: Inventory Intelligence & Uncertainty Buffers**
  - Residual-based variance estimation, service level sensitivity analysis, and preparation recommendations.
- [ ] **Phase 4: Streamlit Dashboard & Portfolio Deployment**
  - Interactive multi-scenario planner, confidence band visualizations, driver waterfall plots, and live decision cockpit.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.