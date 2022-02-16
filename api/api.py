from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello "}

@app.get("/predict")
def predict() :
    model = joblib.load("model.joblib")
    prediction = model.predict(X_test)
    return prediction
