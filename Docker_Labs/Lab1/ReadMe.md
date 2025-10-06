# FastAPI + Docker ML Model (Wine Dataset)

This project trains a **Gradient Boosting Classifier** on the Wine dataset and serves it through a **FastAPI `/predict` endpoint** inside a Docker container.

---

## How to Run

### Build the Docker image
Run this command from the `Lab1` folder:
```bash
docker build -t wine-api:v1 -f Dockerfile .
```

### Run the container
Expose port `8000` so you can access FastAPI locally:
```bash
docker run -p 8000:8000 wine-api:v1
```

When it runs successfully, you’ll see:
```
The model training was successful with accuracy: 0.94
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Test the API

Open your browser and go to:
[http://localhost:8000/docs](http://localhost:8000/docs)

Click on `/predict` → “Try it out” and enter this JSON:
```json
{
  "features": [13.2, 2.77, 2.51, 18.5, 98.0, 2.23, 2.17, 0.26, 1.35, 3.95, 1.02, 3.58, 1235]
}
```

Click **Execute** — you’ll get:
```json
{"prediction": 0}
```

---

## 🧩 Project Files

```
Lab1/
├── src/
│   ├── main.py
│   ├── requirements.txt
│   └── ReadMe.md
├── Dockerfile
└── README.md
```

---

## Tools Used
- Python 3.9  
- FastAPI  
- Scikit-learn  
- Uvicorn  
- Docker  

---
**Lab:** Docker + FastAPI (MLOps)

Changes Made

- Changed the dataset from **Iris** to **Wine** using `load_wine()` from scikit-learn.  
- Replaced **RandomForestClassifier** with **GradientBoostingClassifier** for improved model interpretability and variety.  
- Added **model accuracy evaluation** using `accuracy_score`.  
- Integrated **FastAPI** to serve predictions via a `/predict` API endpoint.  
- Updated **Dockerfile** to run the FastAPI app using **Uvicorn**.  