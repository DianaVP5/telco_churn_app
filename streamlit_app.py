import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------- 1. Entrenar el modelo al iniciar la app ----------

@st.cache_resource
def train_model():
    # Cargar datos (usa el nombre exacto de tu CSV en el repo)
    df = pd.read_csv("Telco-Customer-Churn-ES.csv")

    # Crear variable objetivo numérica
    df["abandono_flag"] = df["Abandono"].apply(lambda x: 1 if x == "Si" else 0)

    # Seleccionamos algunas variables para el modelo
    features = [
        "Contrato",
        "ServicioInternet",
        "MetodoPago",
        "Permanencia",
        "CargosMensuales",
        "CargosTotales",
    ]

    X = df[features]
    y = df["abandono_flag"]

    # Definir columnas numéricas y categóricas
    numericas = ["Permanencia", "CargosMensuales", "CargosTotales"]
    categoricas = ["Contrato", "ServicioInternet", "MetodoPago"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(drop="first"), categoricas),
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=200, random_state=42
    )

    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("modelo", modelo)]
    )

    # Entrenamos con TODOS los datos (solo para predicción)
    pipe.fit(X, y)

    return pipe, features

modelo, features = train_model()

# ---------- 2. Interfaz de usuario ----------

st.title("Predicción de Abandono de Clientes - Telco")

st.write(
    "Ingrese los datos del cliente para estimar la probabilidad de abandono."
)

# Entradas del usuario
Contrato = st.selectbox(
    "Tipo de contrato",
    ["Month-to-month", "One year", "Two year"],
)

ServicioInternet = st.selectbox(
    "Servicio de Internet",
    ["DSL", "Fiber optic", "No"],
)

MetodoPago = st.selectbox(
    "Método de pago",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

Permanencia = st.slider("Permanencia (meses)", 0, 72, 12)
CargosMensuales = st.number_input(
    "Cargos mensuales", min_value=0.0, value=70.0
)
CargosTotales = st.number_input(
    "Cargos totales", min_value=0.0, value=1000.0
)

# Construir el dataframe del cliente
datos_cliente = pd.DataFrame(
    [
        {
            "Contrato": Contrato,
            "ServicioInternet": ServicioInternet,
            "MetodoPago": MetodoPago,
            "Permanencia": Permanencia,
            "CargosMensuales": CargosMensuales,
            "CargosTotales": CargosTotales,
        }
    ]
)

if st.button("Predecir abandono"):
    prob = modelo.predict_proba(datos_cliente)[0][1]

    st.subheader(f"Probabilidad de abandono: {prob:.2%}")

    if prob >= 0.70:
        st.error("🔴 Riesgo ALTO de abandono")
        st.write(
            "👉 Recomendación: ofrecer descuento relevante, migrar a contrato anual o contactar al cliente."
        )
    elif prob >= 0.40:
        st.warning("🟠 Riesgo MEDIO de abandono")
        st.write(
            "👉 Recomendación: enviar promociones personalizadas y mejorar la comunicación."
        )
    else:
        st.success("🟢 Riesgo BAJO de abandono")
        st.write("👉 Recomendación: mantener la estrategia actual y programas de fidelización.")
