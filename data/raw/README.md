# Raw Data Ingestion Guide

This directory stores the uncompressed, immutable raw CSV data files for the DemandIQ demand forecasting and inventory intelligence system.

---

## 1. Dataset Source

- **Challenge:** Food Demand Forecasting Challenge (Kaggle / Analytics Vidhya)
- **Problem Context:** A meal delivery company operates multiple fulfillment centers across various cities. The objective is to forecast weekly demand (`num_orders`) for each center-meal combination so fulfillment teams can plan raw material procurement, staffing, and kitchen preparation without stockouts or food spoilage.
- **Historical Horizon:** Weeks 1 through 145 (in `train.csv`).
- **Future Test Horizon:** Weeks 146 through 155 (in `test.csv`).

---

## 2. Expected Files and Placements

Place the following 4 uncompressed CSV files directly in this directory (`data/raw/`):

```text
data/raw/
├── train.csv
├── test.csv
├── fulfilment_center_info.csv
└── meal_info.csv
```

---

## 3. Detailed File Schemas & Constraints

### 1. `train.csv`
Primary historical transactions table containing weekly order demand.

| Column | Type | Description | Constraint |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique transaction record identifier | Primary Key (unique, non-null) |
| `week` | Integer | Week number of the record | $1 \le \text{week} \le 145$ |
| `center_id` | Integer | Fulfillment center identifier | Foreign Key $\to$ `fulfilment_center_info.center_id` |
| `meal_id` | Integer | Meal option identifier | Foreign Key $\to$ `meal_info.meal_id` |
| `checkout_price` | Float | Final customer price (including discounts/taxes) | $> 0$ |
| `base_price` | Float | Base list price of the meal | $> 0$ |
| `emailer_for_promotion` | Integer | Indicator if promotional email was sent | $\in \{0, 1\}$ |
| `homepage_featured` | Integer | Indicator if meal was featured on website homepage | $\in \{0, 1\}$ |
| `num_orders` | Integer/Float | Target demand: number of orders placed | $\ge 0$ |

### 2. `test.csv`
Out-of-time evaluation / competition test split for forecasting.

| Column | Type | Description | Constraint |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique test record identifier | Primary Key (unique, non-null) |
| `week` | Integer | Week number of the forecast | $146 \le \text{week} \le 155$ |
| `center_id` | Integer | Fulfillment center identifier | Foreign Key $\to$ `fulfilment_center_info.center_id` |
| `meal_id` | Integer | Meal option identifier | Foreign Key $\to$ `meal_info.meal_id` |
| `checkout_price` | Float | Final customer price | $> 0$ |
| `base_price` | Float | Base list price | $> 0$ |
| `emailer_for_promotion` | Integer | Promotional email indicator | $\in \{0, 1\}$ |
| `homepage_featured` | Integer | Homepage featured indicator | $\in \{0, 1\}$ |

### 3. `fulfilment_center_info.csv`
Metadata describing each fulfillment center location and capacity.

| Column | Type | Description | Constraint |
| :--- | :--- | :--- | :--- |
| `center_id` | Integer | Unique fulfillment center identifier | Primary Key (unique, non-null) |
| `city_code` | Integer | Unique city identifier code | Non-null |
| `region_code` | Integer | Geographic region code | Non-null |
| `center_type` | String | Center classification category (`TYPE_A`, `TYPE_B`, `TYPE_C`) | Non-null |
| `op_area` | Float | Operational area of center (in sq km) | $> 0$ |

### 4. `meal_info.csv`
Metadata describing each meal category and cuisine style.

| Column | Type | Description | Constraint |
| :--- | :--- | :--- | :--- |
| `meal_id` | Integer | Unique meal identifier | Primary Key (unique, non-null) |
| `category` | String | Meal category (e.g. Beverages, Rice Bowl, Desert, etc.) | Non-null |
| `cuisine` | String | Cuisine type (e.g. Italian, Indian, Continental, Thai) | Non-null |

---

## 4. How to Download the Data

### Option A: Kaggle Web Download
1. Visit the dataset on Kaggle: [Food Demand Forecasting Challenge](https://www.kaggle.com/datasets/kanncaa1/food-demand-forecasting) (or search for Food Demand Forecasting on Kaggle / Analytics Vidhya).
2. Download and unzip the archive.
3. Move the four CSV files into `data/raw/`.

### Option B: Kaggle CLI
If you have the `kaggle` CLI configured (`~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d kanncaa1/food-demand-forecasting -p data/raw/ --unzip
```

---

## 5. Automated Ingestion & Validation

Once raw CSV files are placed in `data/raw/`, verify their structural schemas and referential integrity by running:

```bash
uv run python -m demandiq.data
```

The validation engine will automatically verify:
- Presence of all required files and columns
- Primary key uniqueness across all tables
- Value domains (non-negative demand, positive prices, binary promotion flags)
- Referential integrity of `center_id` and `meal_id` mapping to metadata tables

---

## 6. Data Governance & Hygiene
- **Zero Raw Data in Version Control:** All `.csv` files inside `data/raw/` are strictly ignored by `.gitignore`.
- **Immutability:** Raw CSV files are read-only inputs. All downstream aggregations, temporal lags, and cleaning are computed programmatically and written to `data/processed/`.
