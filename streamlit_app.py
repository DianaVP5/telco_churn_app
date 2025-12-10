import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# --------- 1. Entrenar el modelo al iniciar la app ----------

@st.cache_resource
def train_model():
    # Cargar tu archivo (usa los nombres reales)
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # ---------- Mapeo DIRECTO a las columnas reales de tu Excel ----------
    col_target = "Abandono"
    col_contract = "Contrato"
    col_internet = "ServicioInternet"
    col_payment = "MetodoPago"
    col_tenure = "Permanencia"
    col_monthly = "CargosMensuales"
    col_total = "CargosTotales"

    # ---------- Crear variable objetivo numérica ----------
    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "sí", "1", "true"] else 0
    )

    # ---------- Features ----------
    features = [
        col_contract,
        col_internet,
        col_payment,
        col_tenure,
        col_monthly,
        col_total,
    ]

    numericas = [col_tenure, col_monthly, col_total]
    categoricas = [col_contract, col_internet, col_payment]

    # Conversión segura de datos
    df[col_total] = pd.to_numeric(df[col_total], errors="coerce").fillna(0)

    X = df[features]
    y = df["abandono_flag"]

    # ---------- Preprocesador ----------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(drop="first"), categoricas),
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("modelo", modelo)]
    )

    pipe.fit(X, y)

    return pipe


modelo = train_model()

# ---------- 2. Interfaz de usuario ----------

st.title("Predicción de Abandono de Clientes - Telco")
st.write("Ingrese los datos del cliente para estimar la probabilidad de abandono.")


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
CargosMensuales = st.number_input("Cargos mensuales", min_value=0.0, value=70.0)
CargosTotales = st.number_input("Cargos totales", min_value=0.0, value=1000.0)


# Crear dataframe con EXACTAMENTE los nombres de tu Excel
datos_cliente = pd.DataFrame([{
    "Contrato": Contrato,
    "ServicioInternet": ServicioInternet,
    "MetodoPago": MetodoPago,
    "Permanencia": Permanencia,
    "CargosMensuales": CargosMensuales,
    "CargosTotales": CargosTotales,
}])


if st.button("Predecir abandono"):

    probs = modelo.predict_proba(datos_cliente)
    classes = list(modelo.classes_)

    if 1 in classes:
        idx = classes.index(1)
        prob = float(probs[0][idx])
    else:
        prob = 0.0

    st.subheader(f"Probabilidad de abandono: {prob:.2%}")
   
    if prob >= 0.70:
        st.error("🔴 Riesgo ALTO de abandono")
        st.write("👉 Recomendación: ofrecer descuento relevante, migrar a contrato anual o contactar al cliente.")
    elif prob >= 0.40:
        st.warning("🟠 Riesgo MEDIO de abandono")
        st.write("👉 Recomendación: enviar promociones personalizadas y mejorar la comunicación.")
    else:
        st.success("🟢 Riesgo BAJO de abandono")
        st.write("👉 Recomendación: mantener la estrategia actual y programas de fidelización.")
