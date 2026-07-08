from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:55:45 2026

@author: Jacob Avila
"""

"""
Pipeline reproducible para:
1) Leer el CSV de precipitación y temperatura (mensual) del Estado de México
2) Calcular PET (Thornthwaite) y SPEI (log-logística 3 parámetros, ajuste por mes calendario)
3) Construir features (lags, acumulados, anomalías)
4) Entrenar modelos (Regresión y Clasificación) para SPEI-3, SPEI-6, SPEI-12
5) Evaluar métricas (RMSE, MAE, NSE, KGE; Accuracy, F1, AUC; Brier; CRPS por muestras)
6) Cuantificar incertidumbre con bootstrap (intervalos 5–95%)
7) Generar tablas y figuras en carpeta ./outputs

Autor: (Jacob Avila / ChatGPT)
Requisitos sugeridos:
- pandas, numpy, matplotlib, scikit-learn, scipy
- opcionales: xgboost, catboost

Ejecución:
    python spei_ml_pipeline.py
o en notebook copiando celdas.

Notas:
- Latitud centroide Edo. México: 19.35°N
- Split temporal:
    Train: 1985-01 a 2015-12
    Val:   2016-01 a 2020-12
    Test:  2021-01 a fin (según datos)
"""



import os
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()

import matplotlib.colors as mcolors

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# -----------------------------
# Configuración
# -----------------------------

DATA_PATH = "dataset/datos conagua precipitación EdoMex.csv"  # <-- archivo ya cargado
OUTPUT_DIR = "./salidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAT_DEG = 19.35  # centroide Edo. México
LAT_RAD = np.deg2rad(LAT_DEG)

SCALES_K = [3, 6, 12]  # SPEI-3, SPEI-6, SPEI-12

TRAIN_END = "2015-12-31"
VAL_END = "2020-12-31"
TEST_START = "2021-01-01"

EPS = 1e-6
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Intentar importar XGBoost / CatBoost (opcionales)
HAS_XGB = False
HAS_CAT = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# Scipy para ajuste log-logístico
HAS_SCIPY = False
try:
    from scipy.stats import fisk  # fisk = log-logística (2p), usaremos 3p con loc
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Función de saneamiento de datos
# -----------------------------
def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convertir cualquier columna object a numérica si es posible
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X

# -----------------------------
# Utilidades: métricas
# -----------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def nse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

def kge(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    mu_y, mu_p = np.mean(y_true), np.mean(y_pred)
    sd_y, sd_p = np.std(y_true, ddof=1), np.std(y_pred, ddof=1)
    if mu_y == 0 or mu_p == 0 or sd_y == 0:
        return np.nan
    beta = mu_p / mu_y
    gamma = (sd_p / mu_p) / (sd_y / mu_y)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))

def crps_from_samples(samples: np.ndarray, y_true: np.ndarray) -> float:
    """
    CRPS estimado a partir de muestras predictivas:
        CRPS = E|X - y| - 0.5 E|X - X'|
    con estimación empírica.
    samples: shape (n, S)
    y_true: shape (n,)
    """
    samples = np.asarray(samples)
    y_true = np.asarray(y_true).reshape(-1, 1)
    n, S = samples.shape
    term1 = np.mean(np.abs(samples - y_true))
    # E|X - X'| aprox. con pares
    # Para eficiencia: usar ordenamiento por fila
    term2_list = []
    for i in range(n):
        s = np.sort(samples[i])
        # Identidad: mean|X-X'| = 2 * sum_{i} (2i-S-1)*x_i / S^2  (para muestra ordenada)
        idx = np.arange(1, S + 1)
        term2 = (2.0 * np.sum((2 * idx - S - 1) * s)) / (S ** 2)
        term2_list.append(term2)
    term2 = np.mean(term2_list)
    return float(term1 - 0.5 * term2)

# ----------------------------------------------------
# Modelos de persistencia
# ----------------------------------------------------
def persistence_model(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.Series:
    """
    Baseline de persistencia:
    predice que el valor actual es igual al valor observado en el periodo anterior.
    Para el primer registro de test, usa el último valor disponible de train.
    """
    y_pred = test[target].shift(1).copy()

    if len(train) == 0:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    last_train_value = train[target].iloc[-1]
    y_pred.iloc[0] = last_train_value

    return y_pred


def climatology_model(train: pd.DataFrame, test: pd.DataFrame, target: str, date_col: str = "PERIODO") -> pd.Series:
    """
    Baseline climatológico mensual:
    predice el promedio histórico del target para cada mes calendario,
    calculado solo con el conjunto de entrenamiento.
    """
    train_local = train.copy()
    test_local = test.copy()

    train_local["month"] = pd.to_datetime(train_local[date_col]).dt.month
    test_local["month"] = pd.to_datetime(test_local[date_col]).dt.month

    monthly_clim = train_local.groupby("month")[target].mean()
    y_pred = test_local["month"].map(monthly_clim)

    return pd.Series(y_pred.values, index=test.index, name=f"{target}_climatology")

# --------------------------------------------------------
# Evaluación de los modelos de regresion
# --------------------------------------------------------
def evaluate_regression_model(y_true: pd.Series, y_pred: pd.Series, model_name: str, scale_k: int) -> dict:
    baseline_models = ["Persistence", "Climatology"]
    return {
        "scale_k": scale_k,
        "model": model_name,
        "model_type": "Baseline" if model_name in baseline_models else "ML",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "n_test": len(y_true)
    }
# -----------------------------
# 1) Lectura y limpieza
# -----------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalizar nombres por si hay espacios
    df.columns = [c.strip().upper() for c in df.columns]
    # PERIODO a datetime
    if "PERIODO" not in df.columns:
        raise ValueError("No se encontró columna PERIODO en el CSV.")
    df["PERIODO"] = pd.to_datetime(df["PERIODO"])
    # Filtrar entidad
    if "ENTIDAD" in df.columns:
        # aceptar variantes con mayúsculas/minúsculas
        mask = df["ENTIDAD"].astype(str).str.strip().str.lower() == "estado de méxico".lower()
        if not mask.any():
            mask = df["ENTIDAD"].astype(str).str.strip().str.lower() == "estado de mexico"
        df = df.loc[mask].copy()
    else:
        # si no hay ENTIDAD, asumir ya es EdoMex
        df = df.copy()

    # Orden temporal
    df = df.sort_values("PERIODO").reset_index(drop=True)

    # Validaciones mínimas
    needed = ["PRECIPITACION", "MEDIA"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")

    # Renombrar a convención interna
    df = df.rename(columns={
        "PRECIPITACION": "P",
        "MINIMA": "TMIN" if "MINIMA" in df.columns else "TMIN",
        "MEDIA": "T",
        "MAXIMA": "TMAX" if "MAXIMA" in df.columns else "TMAX",
    })

    return df


# -----------------------------
# 2) Thornthwaite PET
# -----------------------------

def month_mid_julian_day() -> Dict[int, int]:
    # Día juliano medio por mes (1-12)
    return {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}

def daylength_hours(lat_rad: float, month: int) -> float:
    Jm = month_mid_julian_day()[month]
    delta = 0.409 * np.sin((2 * np.pi * Jm / 365.0) - 1.39)
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    L = (24.0 / np.pi) * omega_s
    return float(L)

def thornthwaite_pet(df: pd.DataFrame, lat_rad: float) -> pd.Series:
    """
    Calcula PET mensual Thornthwaite con corrección por fotoperiodo y días del mes.
    df requiere columnas: PERIODO (datetime), T (°C)
    """
    out = []
    df2 = df.copy()
    df2["YEAR"] = df2["PERIODO"].dt.year
    df2["MONTH"] = df2["PERIODO"].dt.month
    df2["NDAYS"] = df2["PERIODO"].dt.days_in_month

    # Índice térmico anual I_y
    # i_m = (T/5)^1.514 si T>0, else 0
    df2["i_m"] = np.where(df2["T"] > 0, (df2["T"] / 5.0) ** 1.514, 0.0)
    I_by_year = df2.groupby("YEAR")["i_m"].sum().to_dict()

    # Exponente a_y
    a_by_year = {}
    for y, I in I_by_year.items():
        a = (6.75e-7 * I ** 3) - (7.71e-5 * I ** 2) + (1.792e-2 * I) + 0.49239
        a_by_year[y] = a

    # Calcular PET
    for _, row in df2.iterrows():
        y = int(row["YEAR"])
        m = int(row["MONTH"])
        T = float(row["T"])
        I = float(I_by_year[y])
        a = float(a_by_year[y])

        if T <= 0 or I <= 0:
            pet0 = 0.0
        else:
            pet0 = 16.0 * ((10.0 * T / I) ** a)

        Lm = daylength_hours(lat_rad, m)
        nd = float(row["NDAYS"])
        pet = pet0 * (Lm / 12.0) * (nd / 30.0)
        out.append(pet)

    return pd.Series(out, index=df.index, name="PET")


# -----------------------------
# 3) SPEI: log-logística 3p por mes calendario
# -----------------------------

@dataclass
class LogLogistic3PParams:
    c: float   # shape
    loc: float
    scale: float

def fit_loglogistic_3p(x: np.ndarray) -> LogLogistic3PParams:
    """
    Ajuste log-logística 3p usando scipy.stats.fisk (si disponible).
    fisk.fit devuelve (c, loc, scale) para distribución con soporte (loc, inf).
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 15:
        raise ValueError("Muy pocos datos para ajuste estable de log-logística 3p.")
    if not HAS_SCIPY:
        raise ImportError("scipy no está disponible; se requiere para ajuste log-logístico.")
    c, loc, scale = fisk.fit(x)  # MLE
    return LogLogistic3PParams(c=c, loc=loc, scale=scale)

def cdf_loglogistic_3p(x: np.ndarray, params: LogLogistic3PParams) -> np.ndarray:
    if not HAS_SCIPY:
        raise ImportError("scipy no está disponible; se requiere para evaluar CDF log-logística.")
    return fisk.cdf(x, c=params.c, loc=params.loc, scale=params.scale)

def inverse_standard_normal(p: np.ndarray) -> np.ndarray:
    # Φ^{-1}(p)
    if not HAS_SCIPY:
        # Aproximación racional (Acklam) si no hay scipy
        # (para reproducibilidad sin scipy). Aquí mantenemos scipy como preferido.
        raise ImportError("scipy no está disponible para norm.ppf. Instale scipy o implemente ppf.")
    from scipy.stats import norm
    return norm.ppf(p)

def compute_spei(df: pd.DataFrame, k: int) -> pd.Series:
    """
    Calcula SPEI(k) en serie mensual.
    Requiere columnas: PERIODO, P, T, PET
    Ajuste por mes calendario (12 ajustes).
    """
    d = df.copy()
    d["D"] = d["P"] - d["PET"]
    # Acumulado k
    d[f"X{k}"] = d["D"].rolling(window=k, min_periods=k).sum()

    # Ajuste por mes calendario
    spei_vals = np.full(len(d), np.nan)
    params_by_month: Dict[int, LogLogistic3PParams] = {}

    for m in range(1, 13):
        mask = (d["PERIODO"].dt.month == m) & d[f"X{k}"].notna()
        x_m = d.loc[mask, f"X{k}"].values
        if len(x_m) < 20:
            continue
        params = fit_loglogistic_3p(x_m)
        params_by_month[m] = params
        p = cdf_loglogistic_3p(x_m, params)
        p = np.clip(p, EPS, 1 - EPS)
        z = inverse_standard_normal(p)
        spei_vals[mask.values] = z

    s = pd.Series(spei_vals, index=d.index, name=f"SPEI{k}")
    return s


# -----------------------------
# 4) Features para ML
# -----------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["YEAR"] = d["PERIODO"].dt.year
    d["MONTH"] = d["PERIODO"].dt.month

    # Lags (1–6) para P y T
    for lag in range(1, 7):
        d[f"P_lag{lag}"] = d["P"].shift(lag)
        d[f"T_lag{lag}"] = d["T"].shift(lag)

    # Acumulados móviles (3,6,12) para P y aproximación de "calor" con T
    for w in [3, 6, 12]:
        d[f"P_roll{w}"] = d["P"].rolling(w, min_periods=w).sum()
        d[f"T_roll{w}"] = d["T"].rolling(w, min_periods=w).mean()

    # Anomalías mensuales (climatología por mes calendario)
    clim_P = d.groupby("MONTH")["P"].transform("mean")
    clim_T = d.groupby("MONTH")["T"].transform("mean")
    d["P_anom"] = d["P"] - clim_P
    d["T_anom"] = d["T"] - clim_T

    # Si existen TMIN/TMAX en el archivo (algunos CSV los incluyen)
    if "TMIN" in d.columns:
        for lag in range(1, 7):
            d[f"TMIN_lag{lag}"] = d["TMIN"].shift(lag)
        d["TMIN_anom"] = d["TMIN"] - d.groupby("MONTH")["TMIN"].transform("mean")
    if "TMAX" in d.columns:
        for lag in range(1, 7):
            d[f"TMAX_lag{lag}"] = d["TMAX"].shift(lag)
        d["TMAX_anom"] = d["TMAX"] - d.groupby("MONTH")["TMAX"].transform("mean")

    # Variables calendario (seno/coseno)
    d["month_sin"] = np.sin(2 * np.pi * d["MONTH"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["MONTH"] / 12.0)

    return d


# -----------------------------
# 5) Target de clasificación (categorías SPEI)
# -----------------------------

def spei_to_class(spei: pd.Series) -> pd.Series:
    """
    Categorías:
      0: extremadamente seco (<= -2.0)
      1: severamente seco (-2.0, -1.5]
      2: moderadamente seco (-1.5, -1.0]
      3: normal (-1.0, 1.0)
      4: húmedo (>= 1.0)
    """
    x = spei.values
    y = np.full_like(x, fill_value=np.nan, dtype=float)

    y[x <= -2.0] = 0
    y[(x > -2.0) & (x <= -1.5)] = 1
    y[(x > -1.5) & (x <= -1.0)] = 2
    y[(x > -1.0) & (x < 1.0)] = 3
    y[x >= 1.0] = 4

    return pd.Series(y, index=spei.index, name=f"{spei.name}_CLASS")


# -----------------------------
# 6) Split temporal
# -----------------------------

def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    train = d[d["PERIODO"] <= pd.to_datetime(TRAIN_END)].copy()
    val = d[(d["PERIODO"] > pd.to_datetime(TRAIN_END)) & (d["PERIODO"] <= pd.to_datetime(VAL_END))].copy()
    test = d[d["PERIODO"] >= pd.to_datetime(TEST_START)].copy()
    return train, val, test


# -----------------------------
# 7) Modelos
# -----------------------------

def get_models_regression() -> Dict[str, object]:
    models = {}

    # ANN (MLPRegressor)
    models["ANN"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32),
                             activation="relu",
                             alpha=1e-4,
                             learning_rate_init=1e-3,
                             max_iter=500,
                             random_state=RANDOM_SEED))
    ])

    # SVR (RBF)
    models["SVM"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR(C=10.0, gamma="scale", epsilon=0.05, kernel="rbf"))
    ])

    # Random Forest
    models["RF"] = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_SEED, n_jobs=-1
    )

    # XGBoost (si disponible)
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1
        )

    # CatBoost (si disponible)
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=6,
            loss_function="RMSE",
            random_seed=RANDOM_SEED,
            verbose=False
        )

    return models

def get_models_classification() -> Dict[str, object]:
    models = {}

    # ANN
    models["ANN"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=700,
            random_state=RANDOM_SEED
        ))
    ])

    # SVM
    models["SVM"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=10.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=RANDOM_SEED
        ))
    ])

    # Random Forest
    models["RF"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="multi:softprob",
            random_state=RANDOM_SEED,
            n_jobs=1,
            eval_metric="mlogloss"
        )

    # CatBoost
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=RANDOM_SEED,
            verbose=False
        )

    return models


# -----------------------------
# 8) Bootstrap para incertidumbre
# -----------------------------

def bootstrap_predict_regression(model, X_train, y_train, X_test, n_boot: int = 50) -> np.ndarray:
    n = len(X_train)
    preds = []

    X_train = X_train.copy().replace([np.inf, -np.inf], np.nan)
    X_test = X_test.copy().replace([np.inf, -np.inf], np.nan)

    # Convertir todo a numérico
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Imputación simple
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        Xb = X_train_imp.iloc[idx]
        yb = y_train.iloc[idx]

        m = clone_model(model)
        m.fit(Xb, yb)
        preds.append(m.predict(X_test_imp))

    return np.vstack(preds).T

def bootstrap_predict_proba(model, X_train, y_train, X_test, n_boot: int = 50) -> np.ndarray:
    """
    Devuelve muestras predictivas de probas shape (n_test, n_boot, C)
    (promediables para proba final y para intervalos por clase)
    """
    n = len(X_train)
    probas = []
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]
        m = clone_model(model)
        m.fit(Xb, yb)
        probas.append(m.predict_proba(X_test))
    return np.stack(probas, axis=1)  # (n_test, n_boot, C)

def clone_model(model):
    # clon simple para estimadores sklearn/pipelines y boosters
    import copy
    return copy.deepcopy(model)


# -----------------------------
# 9) Gráficas
# -----------------------------

def plot_climate_series(df: pd.DataFrame, outpath: str):
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(df["PERIODO"], df["P"], linewidth=1.0)
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precipitación (mm)")
    ax2 = ax1.twinx()
    ax2.plot(df["PERIODO"], df["T"], linewidth=1.0)
    ax2.set_ylabel("Temperatura media (°C)")
    plt.title("Serie mensual de precipitación y temperatura media")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    plt.close("all")

def plot_spei_series(df: pd.DataFrame, spei_col: str, outpath: str):
    fig = plt.figure()
    plt.plot(df["PERIODO"], df[spei_col], linewidth=1.0)
    plt.axhline(-1.0, linewidth=1.0)
    plt.axhline(-1.5, linewidth=1.0)
    plt.axhline(-2.0, linewidth=1.0)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Fecha")
    plt.ylabel(spei_col)
    plt.title(f"Evolución temporal de {spei_col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    plt.close("all")

def plot_reliability(y_true: np.ndarray, p_pos: np.ndarray, outpath: str, n_bins: int = 10, title: str = ""):
    """
    Reliability diagram binario (para clases extremas, por ejemplo "seco severo o peor").
    y_true: {0,1}
    p_pos: probabilidad predicha
    """
    prob_true, prob_pred = calibration_curve(y_true, p_pos, n_bins=n_bins, strategy="uniform")
    fig = plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1.0)
    plt.plot([0, 1], [0, 1], linewidth=1.0)
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia observada")
    plt.title(title if title else "Curva de confiabilidad")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    plt.close("all")

from statsmodels.graphics.tsaplots import plot_acf

def plot_spei_acf(df, scales, output_dir):
    
    for k in scales:
        spei_col = f"SPEI{k}"
        
        series = df[spei_col].dropna()

        fig = plt.figure()
        plot_acf(series, lags=36)
        
        plt.title(f"Función de Autocorrelación (ACF) para SPEI-{k}")
        plt.xlabel("Rezago (meses)")
        plt.ylabel("Autocorrelación")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Fig_ACF_SPEI{k}.png"), dpi=300)
        plt.close(fig)

def plot_spei_acf_heatmap(df: pd.DataFrame, scales: list, max_lag: int, output_dir: str):
    """
    Genera un heatmap de autocorrelación:
    filas = escalas SPEI
    columnas = rezagos (lags)
    valores = autocorrelación
    """
    acf_matrix = []

    for k in scales:
        col = f"SPEI{k}"
        series = df[col].dropna().values

        acf_vals = []
        for lag in range(1, max_lag + 1):
            if lag >= len(series):
                acf_vals.append(np.nan)
            else:
                s1 = series[:-lag]
                s2 = series[lag:]
                if len(s1) < 2:
                    acf_vals.append(np.nan)
                else:
                    acf_vals.append(np.corrcoef(s1, s2)[0, 1])

        acf_matrix.append(acf_vals)

    acf_matrix = np.array(acf_matrix)

    fig = plt.figure(figsize=(10, 4.8))
    ax = plt.gca()

    im = ax.imshow(
        acf_matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    )

    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"SPEI-{k}" for k in scales])
    ax.set_xticks(range(0, max_lag, 3))
    ax.set_xticklabels([str(i) for i in range(1, max_lag + 1, 3)])

    ax.set_xlabel("Rezago (meses)")
    ax.set_ylabel("escala SPEI")
    ax.set_title("Mapa de calor de Autocorrelación del SPEI a través de escalas temporales")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Autocorrelación")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Fig_ACF_Heatmap_SPEI.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    plt.close("all")

import shap

def plot_shap_summary(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      model_name: str, scale_k: int, output_dir: str,
                      max_display: int = 15):
    """
    Genera un SHAP summary plot para modelos basados en árboles.
    Usa una muestra de test para acelerar el cálculo.
    """
    try:
        # Limpiar datos
        X_train_local = X_train.copy().replace([np.inf, -np.inf], np.nan)
        X_test_local = X_test.copy().replace([np.inf, -np.inf], np.nan)

        for col in X_train_local.columns:
            X_train_local[col] = pd.to_numeric(X_train_local[col], errors="coerce")
            X_test_local[col] = pd.to_numeric(X_test_local[col], errors="coerce")

        # Imputación simple
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train_local),
            columns=X_train_local.columns,
            index=X_train_local.index
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test_local),
            columns=X_test_local.columns,
            index=X_test_local.index
        )

        # Tomar una muestra del test para acelerar SHAP
        if len(X_test_imp) > 100:
            X_shap = X_test_imp.sample(n=100, random_state=RANDOM_SEED)
        else:
            X_shap = X_test_imp.copy()

        # Explainer
        explainer = shap.Explainer(model, X_train_imp)
        shap_values = explainer(X_shap)

        # Summary plot
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_shap,
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Summary Plot - {model_name} - SPEI{scale_k}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"Fig_SHAP_{model_name}_SPEI{scale_k}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close("all")

        # Guardar también importancia media absoluta SHAP en CSV
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame({
            "feature": X_shap.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(
            os.path.join(output_dir, f"Tabla_SHAP_{model_name}_SPEI{scale_k}.csv"),
            index=False
        )

        print(f"[INFO] SHAP generado para {model_name} en SPEI{scale_k}")

    except Exception as e:
        print(f"[WARNING] No se pudo generar SHAP para {model_name} en SPEI{scale_k}: {e}")

# -----------------------------
# 10) Ejecución principal
# -----------------------------

def main():
    # 1) Cargar datos
    df = load_dataset(DATA_PATH)

    # 2) PET y SPEI
    df["PET"] = thornthwaite_pet(df, LAT_RAD)

    # Figuras base: clima
    plot_climate_series(df, os.path.join(OUTPUT_DIR, "Fig1_clima_precip_temp.png"))

    # 3) Calcular SPEI para escalas
    for k in SCALES_K:
        df[f"SPEI{k}"] = compute_spei(df, k)

    # Figuras SPEI
    for k in SCALES_K:
        plot_spei_series(df, f"SPEI{k}", os.path.join(OUTPUT_DIR, f"Fig_SPEI{k}.png"))

    # 4) Features
    df_feat = add_features(df)

    # 5) Preparar datasets por escala k
    results_rows_reg = []
    results_rows_clf = []
    results_rows_prob = []
    
    # -------------------------
    # Graficas ACF del SPEI
    # -------------------------
    plot_spei_acf(df_feat, [3,6,12], OUTPUT_DIR)
    plot_spei_acf_heatmap(df_feat, [3, 6, 12], max_lag=24, output_dir=OUTPUT_DIR)
    
    
    

    for k in SCALES_K:
        target_reg = f"SPEI{k}"
        df_feat[f"{target_reg}_CLASS"] = spei_to_class(df_feat[target_reg])

        # Selección de features
        feature_cols = [c for c in df_feat.columns if c not in
                        ["CVE_ENT", "ENTIDAD", "PERIODO"] and
                        not c.startswith("SPEI") and
                        not c.endswith("_CLASS")]

        # Filtrar filas válidas (sin NaN en target)
        d_k = df_feat[df_feat[target_reg].notna()].copy()

        # Split temporal
        train, val, test = temporal_split(d_k)

        # X/y regresión
        X_train, y_train = train[feature_cols], train[target_reg]
        X_val, y_val = val[feature_cols], val[target_reg]
        X_test, y_test = test[feature_cols], test[target_reg]

        # X/y clasificación
        yc_train = train[f"{target_reg}_CLASS"].dropna().astype(int)
        Xc_train = train.loc[yc_train.index, feature_cols]
        yc_val = val[f"{target_reg}_CLASS"].dropna().astype(int)
        Xc_val = val.loc[yc_val.index, feature_cols]
        yc_test = test[f"{target_reg}_CLASS"].dropna().astype(int)
        Xc_test = test.loc[yc_test.index, feature_cols]
        
        # Saneamiento de variables de entrenamiento, validación y prueba
        X_train = sanitize_features(X_train)
        X_val = sanitize_features(X_val)
        X_test = sanitize_features(X_test)
        
        # Eliminar filas con target faltante
        mask_train = y_train.notna()
        X_train = X_train.loc[mask_train].copy()
        y_train = y_train.loc[mask_train].copy()
        
        mask_val = y_val.notna()
        X_val = X_val.loc[mask_val].copy()
        y_val = y_val.loc[mask_val].copy()
        
        mask_test = y_test.notna()
        X_test = X_test.loc[mask_test].copy()
        y_test = y_test.loc[mask_test].copy()

        # -------------------------
        # Regresión
        # -------------------------
        # ===== Baselines =====
        try:
            y_pred_persistence = persistence_model(train, test, target_reg)
            results_rows_reg.append(
                evaluate_regression_model(y_test, y_pred_persistence, "Persistence", k)
            )
        except Exception as e:
            print(f"[WARNING] Falló baseline Persistence en SPEI{k}: {e}")
        
        try:
            y_pred_climatology = climatology_model(train, test, target_reg, date_col="PERIODO")
            results_rows_reg.append(
                evaluate_regression_model(y_test, y_pred_climatology, "Climatology", k)
            )
        except Exception as e:
            print(f"[WARNING] Falló baseline Climatology en SPEI{k}: {e}")
        
        # ===== Modelos ML =====
        reg_models = get_models_regression()
        
        for name, model in reg_models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                row = evaluate_regression_model(y_test, pred, name, k)
                results_rows_reg.append(row)
                
                # Generar SHAP solo para modelos basados en árboles
                if name in ["RF", "XGBoost", "CatBoost"]:
                    plot_shap_summary(model, X_train, X_test, name, k, OUTPUT_DIR, max_display=15)
        
                # Bootstrap incertidumbre solo para ensembles seleccionados
                if name in ["RF", "XGBoost"]:   # puedes agregar CatBoost si ya te corre estable
                    try:
                        samples = bootstrap_predict_regression(model, X_train, y_train, X_test, n_boot=50)
                        crps = crps_from_samples(samples, y_test.values)
        
                        q05 = np.quantile(samples, 0.05, axis=1)
                        q95 = np.quantile(samples, 0.95, axis=1)
                        coverage = float(np.mean((y_test.values >= q05) & (y_test.values <= q95)))
                        width = float(np.mean(q95 - q05))
        
                        results_rows_prob.append({
                            "scale_k": k,
                            "model": name,
                            "CRPS": crps,
                            "PI90_coverage": coverage,
                            "PI90_width": width,
                            "n_test": len(y_test)
                        })
        
                        nplot = min(200, len(y_test))
                        fig = plt.figure()
                        t_axis = test["PERIODO"].values[:nplot]
                        plt.plot(t_axis, y_test.values[:nplot], linewidth=1.0, label="Observed")
                        plt.plot(t_axis, np.mean(samples, axis=1)[:nplot], linewidth=1.0, label="Predicted")
                        plt.fill_between(t_axis, q05[:nplot], q95[:nplot], alpha=0.3, label="PI90")
                        plt.xlabel("Fecha")
                        plt.ylabel(f"SPEI{k}")
                        plt.title(f"Intervalo predictivo 90% - {name} - SPEI{k}")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_DIR, f"Fig_PI90_{name}_SPEI{k}.png"), dpi=300, bbox_inches="tight")
                        plt.close(fig)
                        plt.close("all")
        
                    except Exception as e:
                        print(f"[WARNING] Falló bootstrap para {name} en SPEI{k}: {e}")
        
            except Exception as e:
                print(f"[WARNING] Falló regresión para {name} en SPEI{k}: {e}")

        # -------------------------
        # Clasificación
        # -------------------------
        clf_models = get_models_classification()
        
        for name, model in clf_models.items():
            try:
                # Verificar número mínimo de clases en entrenamiento
                unique_train_classes = np.unique(yc_train)
                if len(unique_train_classes) < 2:
                    print(f"[WARNING] Muy pocas clases en entrenamiento para {name} en SPEI{k}: {unique_train_classes}")
                    continue
        
                # Codificación contigua de clases para modelos como XGBoost
                le = LabelEncoder()
                yc_train_enc = pd.Series(le.fit_transform(yc_train), index=yc_train.index)
        
                # Filtrar test para conservar sólo clases vistas en entrenamiento
                valid_test_mask = yc_test.isin(le.classes_)
                Xc_test_sub = Xc_test.loc[valid_test_mask].copy()
                yc_test_sub = yc_test.loc[valid_test_mask].copy()
        
                if len(yc_test_sub) == 0:
                    print(f"[WARNING] No hay muestras válidas en test para {name} en SPEI{k}")
                    continue
        
                yc_test_enc = pd.Series(le.transform(yc_test_sub), index=yc_test_sub.index)
        
                # También filtrar validación si decides usarla después
                valid_val_mask = yc_val.isin(le.classes_)
                Xc_val_sub = Xc_val.loc[valid_val_mask].copy()
                yc_val_sub = yc_val.loc[valid_val_mask].copy()
        
                # Asegurar tipos enteros
                yc_train_enc = yc_train_enc.astype(int)
                yc_test_enc = yc_test_enc.astype(int)
        
                # Entrenamiento
                model.fit(Xc_train, yc_train_enc)
        
                # Predicción codificada
                yhat_enc = model.predict(Xc_test_sub)
                yhat_enc = np.asarray(yhat_enc).astype(int)
        
                # Regresar a etiquetas originales
                yhat = le.inverse_transform(yhat_enc)
        
                # Probabilidades
                proba = None
                try:
                    proba = model.predict_proba(Xc_test_sub)
                except Exception:
                    proba = None
        
                # Métricas básicas
                row = {
                    "scale_k": k,
                    "model": name,
                    "Accuracy": float(accuracy_score(yc_test_sub, yhat)),
                    "F1_macro": float(f1_score(yc_test_sub, yhat, average="macro")),
                    "n_test": len(yc_test_sub),
                    "classes_train": ",".join(map(str, le.classes_))
                }
        
                # AUC multiclase sobre etiquetas codificadas
                if proba is not None and len(np.unique(yc_test_enc)) > 1:
                    try:
                        row["AUC_ovr"] = float(
                            roc_auc_score(yc_test_enc, proba, multi_class="ovr")
                        )
                    except Exception as e:
                        print(f"[WARNING] No se pudo calcular AUC para {name} en SPEI{k}: {e}")
                        row["AUC_ovr"] = np.nan
                else:
                    row["AUC_ovr"] = np.nan
        
                results_rows_clf.append(row)
        
                # Brier score para evento binario: sequía severa o peor
                # Solo tiene sentido si existen las clases originales 0 y/o 1
                if proba is not None:
                    try:
                        severe_classes_present = [c for c in [0, 1] if c in le.classes_]
        
                        if len(severe_classes_present) > 0:
                            y_bin = yc_test_sub.isin([0, 1]).astype(int).values
        
                            # Sumar probabilidades de clases 0 y 1 en la codificación original
                            p_bin = np.zeros(len(yc_test_sub), dtype=float)
                            for original_class in severe_classes_present:
                                encoded_class = np.where(le.classes_ == original_class)[0][0]
                                p_bin += proba[:, encoded_class]
        
                            bs = float(brier_score_loss(y_bin, p_bin))
        
                            results_rows_prob.append({
                                "scale_k": k,
                                "model": name,
                                "Brier_severe_or_worse": bs,
                                "n_test": len(y_bin)
                            })
        
                            # Curva de confiabilidad
                            plot_reliability(
                                y_true=y_bin,
                                p_pos=p_bin,
                                outpath=os.path.join(OUTPUT_DIR, f"Fig_Reliability_{name}_SPEI{k}.png"),
                                n_bins=10,
                                title=f"Confiabilidad: P(sequía severa o peor) - {name} - SPEI{k}"
                            )
        
                    except Exception as e:
                        print(f"[WARNING] No se pudo calcular Brier/reliability para {name} en SPEI{k}: {e}")
        
            except Exception as e:
                print(f"[WARNING] Falló clasificación para {name} en SPEI{k}: {e}")
                continue
            
        plt.close("all")
                
    # -----------------------------
    # Guardar tablas
    # -----------------------------
    df_reg = pd.DataFrame(results_rows_reg).sort_values(["scale_k", "KGE", "NSE"], ascending=[True, False, False])
    df_clf = pd.DataFrame(results_rows_clf).sort_values(["scale_k", "F1_macro"], ascending=[True, False])
    df_prob = pd.DataFrame(results_rows_prob)

    df_reg.to_csv(os.path.join(OUTPUT_DIR, "Tabla_Regresion_metricas.csv"), index=False)
    df_clf.to_csv(os.path.join(OUTPUT_DIR, "Tabla_Clasificacion_metricas.csv"), index=False)
    df_prob.to_csv(os.path.join(OUTPUT_DIR, "Tabla_Probabilistica_metricas.csv"), index=False)

    # Resumen en consola
    print("\n=== Modelos disponibles ===")
    print(f"XGBoost instalado: {HAS_XGB}")
    print(f"CatBoost instalado: {HAS_CAT}")
    print(f"Scipy instalado: {HAS_SCIPY}")

    print("\n=== Tabla regresión (top por escala) ===")
    if len(df_reg) > 0:
        print(df_reg.groupby("scale_k").head(3).to_string(index=False))

    print("\n=== Tabla clasificación (top por escala) ===")
    if len(df_clf) > 0:
        print(df_clf.groupby("scale_k").head(3).to_string(index=False))

    print(f"\nArchivos generados en: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
