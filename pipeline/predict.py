import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import Union
from main import model

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

with open("json1.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
       })
with open("json2.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
       })

with open("json3.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
       })

with open("json4.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
       })

with open("json5.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
       })

with open("json6.json") as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print({
            'client_id': {form["client_id"]},
            'Result': y[0]
        })
with open("json7.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
    })

with open("json8.json") as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print({
            'client_id': {form["client_id"]},
            'Result': y[0]
        })

with open("json9.json") as fin:
    form = json.load(fin)
    df = pd.DataFrame.from_dict([form])
    y = model['model'].predict(df)
    print({
        'client_id': {form["client_id"]},
        'Result': y[0]
    })
with open("json10.json") as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print({
            'client_id': {form["client_id"]},
            'Result': y[0]
        })