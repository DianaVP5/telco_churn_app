import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --------- Función auxiliar para buscar columnas por palabra clave ---------
def find_col(df, keywords):
    for col in df.columns:
        name = col.strip().lower().replace(" ", "")
        for kw in keywords:
            if kw in name:
                return col
    return None


# ---------- 1. Entrenar el modelo al iniciar la app ----------
@st.cache_resource
def train_model():

    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Mostrar columnas cargadas
    st.write("📌 Columnas detectadas en el CSV:", list(df.columns))

    # ---------- Detectar columnas reales ----------
    col_target = find_col(df, ["churn", "abandono"])
    if col_target is None:
        raise ValueError("❌ No se encontró columna objetivo (Churn / Abandono)")

    col_contract = find_col(df, ["contract", "contrato"])
    col_internet = find_col(df, ["internetservice", "internet"])
    col_payment = find_col(df, ["paymentmethod", "metodopago", "pago"])
    col_tenure = find_col(df, ["tenure", "permanencia", "meses"])
    col_monthly = find_col(df, ["monthlycharges", "cargosmensuales"])
    col_total = find_col(df, ["totalcharges", "cargostotales", "total"])

    # ---------- Crear columna objetivo ----------
    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "sí", "true", "1"] else 0
    )

    # ---------- Construcción de features ----------
    features = []
    numericas = []
    categoricas = []

    if col_contract:
        features.append(col_contract)
        categoricas.append(col_contract)

    if col_internet:
        features.append(col_internet)
        categoricas.append(col_internet)

    if col_payment:
        features.append(col_payment)
        categoricas.append(col_payment)

    if col_tenure:
        features.append(col_tenure)
        numericas.append(col_tenure)

    if col_monthly:
        features.append(col_monthly)
        numericas.append(col_monthly)

    # ---- Total charges puede NO existir ----
    if col_total is not None and col_total in df.columns:
        df[col_total] = pd.to_numeric(df[col_total], errors="coerce").fillna(0)
        features.append(col_total)
        numericas.append(col_total)
    else:
        st.warning("⚠ No se encontró columna de Cargos Totales. El modelo entrenará sin ella.")
        col_total = None

    # Quitar duplicados
    features = list(dict.fromkeys(features))
    numericas = list(dict.fromkeys(numericas))
    categoricas = list(dict.fromkeys([c for c in categoricas if c not in numericas]))

    # Construir matrices
    X = df[features]
    y = df["abandono_flag"]

    # Preprocesador
    transformers = []
    if len(numericas) > 0:
        transformers.append(("num", StandardScaler(), numericas))
    if len(categoricas) > 0:
        transformers.append(("cat", OneHotEncoder(drop="first"), categoricas))

    preprocessor = ColumnTransformer(transformers=transformers)

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("modelo", modelo)
    ])

    pipe.fit(X, y)

    return pipe, {
        "contract": col_contract,
        "internet": col_internet,
        "payment": col_payment,
        "tenure": col_tenure,
        "monthly": col_monthly,
        "total": col_total
    }


modelo, colnames = train_model()

# ---------- 2. UI ----------
st.title("Predicción de Abandono de Clientes - Telco")
st.write("Ingrese los datos del cliente para estimar la probabilidad de abandono.")

Contrato = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
ServicioInternet = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
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

# Construir DF del cliente
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
        st.write("👉 Ofrecer descuento relevante o plan anual.")
    elif prob >= 0.40:
        st.warning("🟠 Riesgo MEDIO de abandono")
        st.write("👉 Enviar promociones personalizadas.")
    else:
        st.success("🟢 Riesgo BAJO de abandono")
        st.write("👉 Mantener estrategia actual y fidelización.")
