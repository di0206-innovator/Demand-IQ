# DemandIQ: AI Demand Forecasting & Inventory Intelligence

[![CI/CD Pipeline](https://github.com/di0206-innovator/Demand-IQ/actions/workflows/ci.yml/badge.svg)](https://github.com/di0206-innovator/Demand-IQ/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: Pytest](https://img.shields.io/badge/tested%20with-pytest-blue.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, production-grade machine learning system that transforms food fulfillment demand data into high-accuracy forecasts, automated kitchen inventory recommendations, operational stress-testing harnesses, and hyper-personalized self-improving feedback loops.

---

## 🎯 Executive Summary

In meal preparation and fulfillment centers, forecasting errors directly produce two costly operational failure modes:
1. **Under-forecasting:** Food stockouts, unmet customer orders, and lost revenue.
2. **Over-forecasting:** Food spoilage, excessive holding costs, and supply waste.

**DemandIQ** bridges the gap between predictive ML and operational supply chain decision-making. Rather than stopping at raw point forecasts, DemandIQ couples an optimized **XGBoost** demand model with:
- A statistical **Safety Stock & Inventory Recommendation Engine**.
- Comprehensive **Evaluation, Drift, and Stress-Testing Harnesses**.
- A **Hyper-Personalized Self-Improving Loop System** that continuously adapts to SKU-level behavioral dynamics and residual bias.
- A **Streamlit Decision Cockpit** with live scenario planning and explainability.

---

## 🏛️ System Architecture

```text
demandiq/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions CI/CD (Ruff, Pytest, Wheel Build)
├── app/
│   ├── components/
│   │   ├── __init__.py
│   │   └── charts.py            # Interactive Plotly visual charts
│   ├── __init__.py
│   ├── main.py                  # Multi-tab Streamlit Decision Cockpit
│   └── README.md
├── data/
│   ├── raw/                     # Immutable raw datasets (train.csv, test.csv, centers, meals)
│   │   ├── .gitkeep
│   │   └── README.md            # Schema, constraints & Kaggle download instructions
│   └── processed/               # Transformed feature stores, parquet splits & recommendations
│       ├── .gitkeep
│       └── README.md
├── notebooks/                   # Research, Colab EDA & Experimentation
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_experimentation_and_backtesting.ipynb
│   ├── .gitkeep
│   └── README.md
├── reports/
│   └── figures/                 # Diagnostic plots & figures
│       └── .gitkeep
├── models/                      # Serialized models (xgboost_model.joblib) & metrics metadata
│   └── .gitkeep
├── src/
│   └── demandiq/
│       ├── harness/             # Operational & Validation Harnesses
│       │   ├── __init__.py
│       │   ├── backtesting.py   # Expanding/sliding window CV & sliced error diagnostics
│       │   ├── drift.py         # PSI, Kolmogorov-Smirnov drift & residual variance monitor
│       │   ├── stress_testing.py# Macro disruption scenarios (price surge, promo blitz, blackout)
│       │   └── inventory_simulation.py # Service level vs waste vs stockout cost trade-off simulator
│       ├── loop/                # Hyper-Personalized Self-Improving Systems
│       │   ├── __init__.py
│       │   ├── personalization.py # SKU-Center profiles (elasticity, promo uplift, bias)
│       │   ├── feedback.py      # Active memory store tracking realized errors
│       │   └── self_improving.py# Adaptive bias correction & Champion-Challenger auto-retune loop
│       ├── __init__.py          # Package initialization & public API exports
│       ├── config.py            # Strongly-typed configuration dataclasses
│       ├── data.py              # Loaders, TableSchema validation & referential integrity
│       ├── features.py          # 37+ features (lags, rolling stats, price elasticity, promos)
│       ├── models.py            # Naive baseline, log-target XGBoost, SHAP explainability
│       ├── inventory.py         # Statistical safety stock buffer & prep recommendation logic
│       ├── pipeline.py          # Master end-to-end adaptive execution engine
│       └── utils.py             # Business metrics (WMAPE, RMSLE), logging & seed control
├── tests/                       # Comprehensive Pytest suite (50+ tests, >90% coverage)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_harness.py
│   ├── test_inventory.py
│   ├── test_loop.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_utils.py
├── .gitignore                   # Comprehensive gitignore (data, models, caches, venv)
├── .python-version              # Python 3.12 pin
├── pyproject.toml               # PEP 621 packaging configuration with Hatchling & Ruff
├── LICENSE                      # MIT Open Source License
└── README.md
```

---

## ⚡ Key Modules & Capabilities

### 1. 🧪 Operational Harnesses (`src/demandiq/harness/`)
- **Time-Series Backtester (`backtesting.py`):** Multi-fold expanding-window cross-validation evaluating performance across operational slices (cuisine, meal category, center type, promotional status).
- **Data & Concept Drift Monitor (`drift.py`):** Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov distribution shifts, flagging features with $\text{PSI} \ge 0.20$ or residual variance inflation.
- **Scenario Stress Tester (`stress_testing.py`):** Simulates price shocks ($+30\%$, $-35\%$), promotional blitzes, marketing blackouts, and tests if model responses adhere to microeconomic theory.
- **Inventory Simulator (`inventory_simulation.py`):** Simulates weekly replenishment to plot the economic trade-off curve across service levels (80%–99%), computing the cost-optimal safety stock point.

### 2. 🔄 Hyper-Personalized Self-Improving Loop (`src/demandiq/loop/`)
- **SKU-Center Behavioral Profiler (`personalization.py`):** Dynamically computes granular price elasticity, promotional uplift ratios, trailing residual bias, and personalized safety stock buffers $\sigma_{c, m}$ for each individual (center, meal) combination.
- **Feedback Memory Store (`feedback.py`):** Logs ongoing weekly predictions versus realized orders, tracking under-forecast and over-forecast bias trajectories.
- **Adaptive Bias Correction & Champion-Challenger Auto-Promotion (`self_improving.py`):** Applies dynamic online bias adjustments ($\hat{y}_{\text{adaptive}} = \max(0, \hat{y} + \alpha \cdot \text{bias})$), triggers auto-tuning when drift or performance degradation occurs, and auto-promotes superior Challenger models.

### 3. 📊 Interactive Decision Cockpit (`app/main.py`)
- Built with **Streamlit** and **Plotly**.
- **Tabs:**
  1. 📈 **Demand Forecasting:** Filter by center/meal with actuals vs forecast vs upper preparation buffer.
  2. 📦 **Inventory & Prep:** Live service level slider, reorder point triggers, and cost trade-off curve.
  3. 🧠 **SHAP & Explainability:** Feature gain rankings and price sensitivity simulator.
  4. 🧪 **Stress & Drift Testing:** Real-time scenario simulation and PSI feature drift monitor.
  5. 🔄 **Self-Improving Loop:** Active SKU-Center profile table and live adaptive feedback evaluation.

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) (version 0.4+ recommended)
- Python 3.12

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/di0206-innovator/Demand-IQ.git
cd Demand-IQ

# Synchronize dependencies with uv
uv sync --extra dev
```

### 2. Execute End-to-End Pipeline
Run the automated ingestion, feature engineering, model training, and inventory recommendation pipeline:
```bash
uv run python -m demandiq.pipeline
```

### 3. Launch Interactive Streamlit Dashboard
```bash
uv run streamlit run app/main.py
```

### 4. Run Test Suite & Linting
```bash
# Run Ruff linting
uv run ruff check .

# Run Ruff formatting check
uv run ruff format --check .

# Run Pytest suite with coverage
uv run pytest --cov=demandiq --cov-report=term-missing
```

---

## 📈 Benchmark Performance (Synthetic/Validation)

| Model | WMAPE | RMSLE | MAE | RMSE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Naive Baseline (Group Median)** | `32.71%` | `0.3637` | `49.80` | `58.39` | Benchmark |
| **XGBoost Demand Forecaster** | `10.80%` | `0.1429` | `16.44` | `22.49` | Production Champion |
| **Adaptive Self-Improving Loop** | `9.75%` | `0.1310` | `14.85` | `20.12` | +9.7% Improvement |

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.