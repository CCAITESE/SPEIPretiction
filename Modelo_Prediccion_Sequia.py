# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:18:39 2026

@author: Jacob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv("dataset/datos conagua precipitación EdoMex.csv")

## Verificamos valores nulos
datos.info()
datos.isnull().sum()
datos.isna().sum()

columnas = datos.columns
print (columnas)

## Estadísticos descriptivos
descriptivos = datos.describe()

## Convertir periodo a fecha y colocarlo como índice
datos['PERIODO'] = pd.to_datetime(datos['PERIODO'])
datos.set_index('PERIODO', inplace=True)

datos.groupby("PERIODO")['MEDIA'].mean()

plt.plot(datos.index, datos['MEDIA'])
plt.xlabel("Periodo")
plt.ylabel("Temperatura media")
plt.show()

precip_prom = datos["PRECIPITACION"].mean()
temp_prom = datos["MEDIA"].mean()

print("Precipitación promedio:", precip_prom)
print("Temperatura media promedio:", temp_prom)

## Para analizar tendencias y estacionalidad mensual, la descomposición es clave
## 
from statsmodels.tsa.seasonal import seasonal_decompose

## # Descomposición de la serie (period=12 para datos mensuales)
resultado = seasonal_decompose(datos['MEDIA'], model='additive', period=12)
resultado.plot()
plt.show()

## Resampling (Remuestreo): Convertir datos diarios a mensuales
# Si tuvieras datos diarios y quieres mensual 'mean'
# df_mensual = df_diario.resample('M').mean()


## Shifting (Desplazamiento): Comparar el mes actual con el anterior
datos['anterior'] = datos['MEDIA'].shift(1)


## Modelado y Pronóstico (Forecasting)
## Para predecir valores futuros, el modelo ARIMA/SARIMA es muy común
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ajustar un modelo ARIMA (p,d,q)
model = SARIMAX(datos['MEDIA'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Pronosticar los próximos 6 meses
forecast = results.get_forecast(steps=6)
print(forecast.predicted_mean)

plt.bar(datos.index, datos['PRECIPITACION'])
plt.plot(datos.index, datos['MEDIA'], color='red')
plt.show()
