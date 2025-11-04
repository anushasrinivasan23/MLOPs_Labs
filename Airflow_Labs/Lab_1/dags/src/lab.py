import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os


def load_and_validate_data():
    """
    Loads data from CSV file, validates it, and returns serialized data.
    
    Returns:
        bytes: Serialized validated data.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    
    # Data validation
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Validate required columns exist
    required_cols = ['size', 'bedrooms', 'bathrooms', 'age', 'location_score', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Validated data shape: {df.shape}")
    serialized_data = pickle.dumps(df)
    return serialized_data


def feature_engineering(data):
    """
    Performs feature engineering on the data.
    
    Args:
        data (bytes): Serialized data.
        
    Returns:
        bytes: Serialized engineered features.
    """
    df = pickle.loads(data)
    
    # Create new features
    df['size_per_bedroom'] = df['size'] / (df['bedrooms'] + 1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / (df['size'] + 1)
    df['age_category'] = pd.cut(df['age'], bins=[-1, 10, 25, 50, 100], labels=['new', 'modern', 'old', 'very_old'])
    df['location_score_category'] = pd.cut(df['location_score'], bins=[0, 3, 6, 8, 10], labels=['poor', 'fair', 'good', 'excellent'])
    
    # Select features for modeling
    feature_cols = ['size', 'bedrooms', 'bathrooms', 'age', 'location_score', 
                    'size_per_bedroom', 'total_rooms']
    X = df[feature_cols]
    y = df['price']
    
    print(f"Feature engineering complete. Features: {feature_cols}")
    print(f"Target variable: price")
    
    result = {'X': X, 'y': y, 'feature_cols': feature_cols}
    return pickle.dumps(result)


def train_model(data, filename):
    """
    Trains a Random Forest regression model and saves it.
    
    Args:
        data (bytes): Serialized feature-engineered data.
        filename (str): Model filename.
        
    Returns:
        dict: Model metrics and scaler.
    """
    data_dict = pickle.loads(data)
    X = data_dict['X']
    y = data_dict['y']
    feature_cols = data_dict['feature_cols']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # Save model and scaler
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, filename)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_cols': feature_cols
    }
    return pickle.dumps(metrics)


def make_predictions(filename, metrics_data):
    """
    Makes predictions on test data using the trained model.
    
    Args:
        filename (str): Model filename.
        metrics_data (bytes): Serialized metrics data.
        
    Returns:
        dict: Prediction results.
    """
    metrics = pickle.loads(metrics_data)
    feature_cols = metrics['feature_cols']
    
    # Load model and scaler
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    scaler_path = os.path.join(os.path.dirname(__file__), "../model", "scaler.pkl")
    
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    
    # Feature engineering on test data
    test_df['size_per_bedroom'] = test_df['size'] / (test_df['bedrooms'] + 1)
    test_df['total_rooms'] = test_df['bedrooms'] + test_df['bathrooms']
    
    # Select features
    X_test = test_df[feature_cols]
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['predicted_price'] = predictions
    
    print(f"Made predictions on {len(test_df)} test samples")
    print(f"Predicted price range: ${predictions.min():,.2f} - ${predictions.max():,.2f}")
    print(f"Average predicted price: ${predictions.mean():,.2f}")
    
    # Save predictions
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    results_path = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")
    
    return pickle.dumps({
        'predictions': predictions.tolist(),
        'avg_prediction': float(predictions.mean())
    })
