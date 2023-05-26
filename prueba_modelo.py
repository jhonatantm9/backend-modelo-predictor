from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Carga del modelo previamente entrenado
model = joblib.load("modelo/modelo_rfr.pkl")

# Carga de los objetos MinMaxScaler previamente entrenados
scaler_input = joblib.load("modelo/minMax_scaler_input.pkl")
scaler_output = joblib.load("modelo/minMax_scaler_output.pkl")

def predict(input_data):
    # Convertir los datos de entrada a un DataFrame
    input_df = pd.DataFrame(input_data)

    # Aplicar las transformaciones necesarias
    scaled_input = scaler_input.transform(input_df)
    log_scaled_input = np.log(scaled_input)

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(log_scaled_input)

    # Aplicar las transformaciones inversas a la salida
    squared_output = np.square(prediction)
    inverse_scaled_output = scaler_output.inverse_transform(squared_output.reshape(1, -1))

    return inverse_scaled_output


datos = [[10000, 400, 5000, 100000, 600000, 30000, 700000, 800000, 1],
         [500000, 20000, 55000, 1030000, 600000, 30000, 700000, 800000, 1.3]]
print(predict(datos))