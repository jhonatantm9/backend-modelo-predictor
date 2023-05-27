import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("modelo/modelo_rfr.pkl")
scaler_input = joblib.load("modelo/minMax_scaler_input.pkl")
scaler_output = joblib.load("modelo/minMax_scaler_output.pkl")

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#modelo = joblib.load('modelo/modelo.pkl')


class Entrada(BaseModel):
    datos: List[
        List[float]
    ]


@app.get("/")
def index():
    return {"message": "Hola mundo"}


@app.post("/predict")
def predict(input_data: Entrada):
    # Convertir los datos de entrada a un DataFrame
    print('Datos de entrada: ' + str(input_data.datos))
    input_df = pd.DataFrame(input_data.datos)
    print('Datos convertidos a dataframe, shape: ' + str(input_df.shape))

    # Aplicar las transformaciones necesarias
    scaled_input = scaler_input.transform(input_df)
    log_scaled_input = np.log(scaled_input)

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(log_scaled_input)

    # Aplicar las transformaciones inversas a la salida
    squared_output = np.square(prediction)
    inverse_scaled_output = scaler_output.inverse_transform(squared_output.reshape(1, -1))

    return {"resultado": inverse_scaled_output.tolist()}
