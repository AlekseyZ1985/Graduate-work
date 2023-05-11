import json
import uvicorn as uvicorn
import joblib
import pandas as pd
import dill
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

app = FastAPI()
with open('sber_pipe.pkl', 'rb') as file:
    model = dill.load(file)
print(model["metadata"])


class Form(BaseModel):
    session_id: Union[str, None] = None
    client_id: Union[str, None] = None
    visit_date: float
    visit_time: Union[str, None] = None
    visit_number: float
    utm_source: Union[str, None] = None
    utm_medium: Union[str, None] = None
    utm_campaign: Union[str, None] = None
    utm_adcontent: Union[str, None] = None
    utm_keyword: Union[str, None] = None
    device_category: Union[str, None] = None
    device_os: Union[str, None] = None
    device_brand: Union[str, None] = None
    device_model: Union[str, None] = None
    device_screen_resolution: Union[str, None] = None
    device_browser: Union[str, None] = None
    geo_country: Union[str, None] = None
    geo_city: Union[str, None] = None

class Prediction(BaseModel):
    client_id: str
    Result: float


@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)

def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'client_id': form.client_id,
        'Result': y[0]
    }