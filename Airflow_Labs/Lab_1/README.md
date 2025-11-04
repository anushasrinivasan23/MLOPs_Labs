# Airflow Lab - House Price Prediction

## Changes Made

### Model Changes
- **Changed from**: K-Means clustering (customer segmentation)
- **Changed to**: Random Forest regression (house price prediction)

### Data Changes
- **Old data**: Customer financial data (BALANCE, PURCHASES, CREDIT_LIMIT)
- **New data**: House property data (`file.csv` and `test.csv`)
  - Features: `size`, `bedrooms`, `bathrooms`, `age`, `location_score`
  - Target: `price` (only in training data)

### Function Changes
Replaced 4 functions with new implementations:

1. **`load_and_validate_data()`** (replaces `load_data()`)
   - Added data validation
   - Checks required columns
   - Handles missing values

2. **`feature_engineering()`** (replaces `data_preprocessing()`)
   - Creates derived features: `size_per_bedroom`, `total_rooms`, `age_category`, `location_score_category`
   - Uses StandardScaler instead of MinMaxScaler

3. **`train_model()`** (replaces `build_save_model()`)
   - Random Forest regressor instead of K-Means
   - Includes train-test split
   - Evaluates with RMSE, MAE, R² metrics
   - Saves both model and scaler

4. **`make_predictions()`** (replaces `load_model_elbow()`)
   - Makes predictions on test data
   - Saves results to `predictions.csv`

## Output Files
- `dags/model/house_price_model.pkl` - Trained Random Forest model
- `dags/model/scaler.pkl` - Feature scaler
- `dags/model/predictions.csv` - Predictions on test data

## Quick Start

1. **Start Airflow**:
   ```bash
   docker compose up
   ```

2. **Access UI**: http://localhost:8080
   - Username: `airflow2`
   - Password: `airflow2`

3. **Trigger DAG**: Find `House_Price_Prediction` and click "Trigger DAG"

4. **View Results**: Check `dags/model/predictions.csv` after execution
