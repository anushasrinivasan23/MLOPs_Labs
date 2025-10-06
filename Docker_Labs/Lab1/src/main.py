# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Wine Model API")

MODEL_PATH = "wine_model.pkl"

# Define input schema for prediction requests
class WineInput(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Wine model API is running! Use /predict endpoint for inference."}

@app.post("/predict")
def predict(data: WineInput):
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model file not found. Please train the model first."}

    model = joblib.load(MODEL_PATH)
    # Convert input features into array and reshape
    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    return {"prediction": prediction}

# Train the model when container first runs
if __name__ == "__main__":
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)

    print(f"The model training was successful with accuracy: {acc:.2f}")

    # Start FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
