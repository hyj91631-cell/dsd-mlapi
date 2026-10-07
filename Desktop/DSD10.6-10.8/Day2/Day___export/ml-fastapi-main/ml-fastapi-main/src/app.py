from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from src.model_registry import retrieve
from src.config import appconfig

app = FastAPI()

model, features = retrieve(appconfig['Model']['name'])

@app.get(appconfig['API']['home'])
def home():
    return {"message": "Welcome to DSSI API v1.0!"}

@app.post(appconfig['API']['used_car_price'])
def predict(data: dict):
    """
    Endpoint to retrieve the predicted price given input data from request.
        Parameters:
            data (dict): Input data from request
        Returns:
            JSON: "price" as key and its predicted value
    """
    pred_df = pd.DataFrame.from_dict([data])
    pred = model.predict(pred_df[features])
    return {"price": pred[0]}
