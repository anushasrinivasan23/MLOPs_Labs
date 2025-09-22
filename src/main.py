# src/main.py
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

class WineResponse(BaseModel):
    class_index: int
    class_name: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        # Arrange features in the same order as dataset
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315,
            wine_features.proline
        ]]

        # Get prediction
        prediction = predict_data(features)

        # Wine dataset has 3 classes
        class_names = ["Barolo", "Grignolino", "Barbera"]

        return WineResponse(
            class_index=int(prediction[0]),
            class_name=class_names[int(prediction[0])]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
