# Raw Data Directory

This directory stores the uncompressed, immutable raw data files for the DemandIQ project.

## Dataset Source
- **Dataset:** Food Demand Forecasting Challenge (Kaggle / Analytics Vidhya)
- **Problem Context:** Weekly meal demand forecasting across 77 fulfillment centers and 51 meal options over 145 weeks.

## Expected Files
When downloading the dataset, place the following CSV files into this directory:

1. `train.csv`:
   - `id`: Unique record identifier
   - `week`: Week number (1 to 145)
   - `center_id`: Unique identifier for fulfillment center
   - `meal_id`: Unique identifier for meal
   - `checkout_price`: Final price charged to customer (including discounts/taxes)
   - `base_price`: Base price of the meal
   - `emailer_for_promotion`: Flag indicating promotional email campaign (0 or 1)
   - `homepage_featured`: Flag indicating meal featured on homepage (0 or 1)
   - `num_orders`: Target demand (number of orders)

2. `fulfilment_center_info.csv`:
   - `center_id`: Unique fulfillment center ID
   - `city_code`: City identifier
   - `region_code`: Region identifier
   - `center_type`: Type of center (TYPE_A, TYPE_B, TYPE_C)
   - `op_area`: Operational area in square km

3. `meal_info.csv`:
   - `meal_id`: Unique meal ID
   - `category`: Food category (e.g., Beverages, Rice Bowl, Desert, etc.)
   - `cuisine`: Cuisine type (e.g., Indian, Italian, Continental, Thai)

4. `test.csv` (Optional evaluation/competition test split):
   - Contains identical schema as `train.csv` except for `num_orders`.

## Data Governance & Reproducibility Notice
- Files in this directory are strictly ignored by Git (`.gitignore`).
- Raw data must remain read-only and never be modified in-place.
- Any cleaning, aggregation, or transformations are handled downstream in `src/demandiq/data.py` and output to `data/processed/`.
