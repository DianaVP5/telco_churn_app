import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# --------- Función auxiliar para buscar columnas por palabra clave ---------
def find_col(df, keywords):
    """
    Detecta columnas incluso si tienen espacios, BOM, mayúsculas o variaciones.
    """
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "")
        for kw in keywords:
            if kw in clean:
                return col
    return None


# ---------- 1. Entrenar el modelo al iniciar la app ----------
@st.cache_resource
def train_model():
    # Cargar datos
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # ---------- Limpiar nombres de columnas ----------
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)

    # Mostrar columnas detectadas
    st.write("Columnas del CSV:", list(df.columns))

    # ---------- Detectar nombres reales de columnas ----------
    col_target = find_col(df, ["churn", "abandono"])
    if col_target is None:
        raise ValueError(
            f"No se encontró la columna objetivo. Columnas: {list(df.columns)}"
        )

    st.write("Columna objetivo detectada:", col_target)

    # Limpiar valores del target (Yes/No)
    df[col_target] = df[col_target].astype(str).str.strip()

    # Mapeo robusto
    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "sí", "1", "true"] else 0
    )

    st.write("Conteo de clases target:", df["abandono_flag"].value_counts())

    # Detectar features
    col_contract = find_col(df, ["contract"])
    col_internet = find_col(df, ["internetservice"])
    col_payment = find_col(df, ["paymentmethod"])
    col_tenure = find_col(df, ["tenure"])
    col_monthly = find_col(df, ["monthlycharges"])
    col_total = find_col(df, ["totalcharges"])

    # Convertir total charges a número
    if col_total:
        df[col_total] = pd.to_numeric(df[col_total], errors="coerce").fillna(0)

    # Features válidas
    numericas = [c for c in [col_tenure, col_monthly, col_total] if c]
    categoricas = [c for c in [col_contract, col_internet, col_payment] if c]
    features = numericas + categoricas

    X = df[features]
    y = df["abandono_flag"]

    # ---------- Preprocesamiento ----------
    transformers = []
    if numericas:
        transformers.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformers.append(("cat", OneHotEncoder(drop="first"), categoricas))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("modelo", modelo)]
    )

    pipe.fit(X, y)

    return pipe, {
        "contract": col_contract,
        "internet": col_internet,
        "payment": col_payment,
        "tenure": col_tenure,
        "monthly": col_monthly,
        "total": col_total,
    }


modelo, colnames = train_model()


# ---------- 2. Interfaz ----------
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


# Construir DF para predicción
datos_dict = {}

if colnames["contract"]: datos_dict[colnames["contract"]] = Contrato
if colnames["internet"]: datos_dict[colnames["internet"]] = ServicioInternet
if colnames["payment"]: datos_dict[colnames["payment"]] = MetodoPago
if colnames["tenure"]: datos_dict[colnames["tenure"]] = Permanencia
if colnames["monthly"]: datos_dict[colnames["monthly"]] = CargosMensuales
if colnames["total"]: datos_dict[colnames["total"]] = CargosTotales

datos_cliente = pd.DataFrame([datos_dict])


# ---------- PREDICCIÓN ----------
if st.button("Predecir abandono"):
    probs = modelo.predict_proba(datos_cliente)
    classes = list(modelo.classes_)

    if 1 in classes:
        prob = float(probs[0][classes.index(1)])
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
