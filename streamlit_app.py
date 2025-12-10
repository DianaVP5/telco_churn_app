import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# --------- Buscar columnas por palabra clave ----------
def find_col(df, keywords):
    """
    Busca una columna que contenga alguna de las keywords en su nombre.
    Ej: keywords=["churn"] encuentra "Churn" o "customer_churn".
    """
    for col in df.columns:
        clean = col.lower().replace(" ", "")
        for kw in keywords:
            if kw in clean:
                return col
    return None


# ---------- Entrenamiento ----------
@st.cache_resource
def train_model():

    df = pd.read_csv("Telco-Customer-Churn.csv")

    # ---------- Detectar columnas ----------
    col_target = find_col(df, ["churn", "abandono"])   # Churn
    col_contract = find_col(df, ["contract"])
    col_internet = find_col(df, ["internetservice"])
    col_payment = find_col(df, ["paymentmethod"])
    col_tenure = find_col(df, ["tenure"])
    col_monthly = find_col(df, ["monthlycharges"])
    col_total = find_col(df, ["totalcharges"])

    # ---------- Procesar columnas ----------
    df[col_total] = pd.to_numeric(df[col_total], errors="coerce").fillna(0)

    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "sí", "1", "true"] else 0
    )

    # ---------- Definir features ----------
    numericas = [col_tenure, col_monthly, col_total]
    numericas = [c for c in numericas if c is not None]

    categoricas = [col_contract, col_internet, col_payment]
    categoricas = [c for c in categoricas if c is not None]

    features = numericas + categoricas

    X = df[features]
    y = df["abandono_flag"]

    # ---------- Preprocesamiento ----------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(drop="first"), categoricas),
        ],
        remainder="drop"
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", modelo)
    ])

    pipeline.fit(X, y)

    return pipeline, {
        "contract": col_contract,
        "internet": col_internet,
        "payment": col_payment,
        "tenure": col_tenure,
        "monthly": col_monthly,
        "total": col_total,
    }


# Entrenar el modelo
modelo, colnames = train_model()


# ---------- STREAMLIT UI ----------
st.title("Predicción de Abandono de Clientes - Telco")
st.write("Ingrese los datos del cliente para estimar la probabilidad de abandono.")


# ---------- Inputs ----------
Contrato = st.selectbox(
    "Tipo de contrato",
    ["Month-to-month", "One year", "Two year"]
)

ServicioInternet = st.selectbox(
    "Servicio de Internet",
    ["DSL", "Fiber optic", "No"]
)

MetodoPago = st.selectbox(
    "Método de pago",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

Permanencia = st.slider("Permanencia (meses)", 0, 72, 5)
CargosMensuales = st.number_input("Cargos mensuales", min_value=0.0, value=75.0)
CargosTotales = st.number_input("Cargos totales", min_value=0.0, value=200.0)


# ---------- Construir dataframe ----------
datos_dict = {}

if colnames["contract"]:
    datos_dict[colnames["contract"]] = Contrato
if colnames["internet"]:
    datos_dict[colnames["internet"]] = ServicioInternet
if colnames["payment"]:
    datos_dict[colnames["payment"]] = MetodoPago
if colnames["tenure"]:
    datos_dict[colnames["tenure"]] = Permanencia
if colnames["monthly"]:
    datos_dict[colnames["monthly"]] = CargosMensuales
if colnames["total"]:
    datos_dict[colnames["total"]] = CargosTotales

datos_cliente = pd.DataFrame([datos_dict])


# ---------- Predicción ----------
if st.button("Predecir abandono"):

    prob = float(modelo.predict_proba(datos_cliente)[0][1])

    st.subheader(f"Probabilidad de abandono: {prob:.2%}")

    # Barra de progreso visual
    st.progress(prob)

    if prob >= 0.70:
        st.error("🔴 Riesgo ALTO de abandono")
        st.write("👉 Ofrecer descuento fuerte, migración anual, llamada inmediata.")
    elif prob >= 0.40:
        st.warning("🟠 Riesgo MEDIO de abandono")
        st.write("👉 Enviar promociones personalizadas, mejorar soporte.")
    else:
        st.success("🟢 Riesgo BAJO de abandono")
        st.write("👉 Mantener estrategia actual, programas de fidelización.")
