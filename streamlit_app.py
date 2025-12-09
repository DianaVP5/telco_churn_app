import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# --------- Función auxiliar para buscar columnas por palabra clave ---------
def find_col(df, keywords):
    """
    Busca una columna que contenga alguna de las keywords en su nombre (ignorando mayúsculas/minúsculas y espacios).
    Ej: keywords=["churn","abandono"] encontrará "Churn", "abandono_cliente", "Churn ".
    """
    cols = list(df.columns)
    for col in cols:
        name = col.strip().lower()
        for kw in keywords:
            if kw in name:
                return col
    return None


# ---------- 1. Entrenar el modelo al iniciar la app ----------

@st.cache_resource
def train_model():
    # Cargar datos (usa el nombre exacto de tu CSV en el repo)
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Si quieres ver las columnas reales, descomenta esta línea:
    # st.write("Columnas del CSV:", list(df.columns))

    # ---------- Detectar nombres reales de columnas ----------

    # Columna objetivo: algo que contenga "churn" o "abandono"
    col_target = find_col(df, ["churn", "abandono"])
    if col_target is None:
        raise ValueError(
            f"No se encontró la columna de abandono (que contenga 'churn' o 'abandono'). "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Contrato
    col_contract = find_col(df, ["contract", "contrato"])

    # Servicio de Internet
    col_internet = find_col(df, ["internetservice", "serviciointernet", "internet"])

    # Método de pago
    col_payment = find_col(df, ["paymentmethod", "metodopago", "pago"])

    # Permanencia
    col_tenure = find_col(df, ["tenure", "permanencia", "meses"])

    # Cargos mensuales
    col_monthly = find_col(df, ["monthlycharges", "cargosmensuales", "mensual"])

    # Cargos totales (opcional)
    col_total = find_col(df, ["totalcharges", "cargostotales", "total"])

    # ---------- Crear variable objetivo numérica ----------

    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "sí", "1", "true"] else 0
    )

    # ---------- Construir lista de features según lo que exista ----------

    features = []
    categoricas = []
    numericas = []

    if col_contract is not None:
        features.append(col_contract)
        categoricas.append(col_contract)

    if col_internet is not None:
        features.append(col_internet)
        categoricas.append(col_internet)

    if col_payment is not None:
        features.append(col_payment)
        categoricas.append(col_payment)

    if col_tenure is not None:
        features.append(col_tenure)
        numericas.append(col_tenure)

    if col_monthly is not None:
        features.append(col_monthly)
        numericas.append(col_monthly)

    if col_total is not None:
        features.append(col_total)
        numericas.append(col_total)
        df[col_total] = pd.to_numeric(df[col_total], errors="coerce").fillna(0)

    if len(features) == 0:
        raise ValueError(
            f"No se encontraron columnas de entrada válidas. Columnas del CSV: {list(df.columns)}"
        )

    X = df[features]
    y = df["abandono_flag"]

    # ---------- Preprocesador ----------

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas) if len(numericas) > 0 else ("num", "passthrough", []),
            ("cat", OneHotEncoder(drop="first"), categoricas) if len(categoricas) > 0 else ("cat", "passthrough", []),
        ],
        remainder="drop",
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("modelo", modelo)]
    )

    pipe.fit(X, y)

    # Devolvemos también los nombres de columnas para construir el DF del cliente
    return pipe, {
        "contract": col_contract,
        "internet": col_internet,
        "payment": col_payment,
        "tenure": col_tenure,
        "monthly": col_monthly,
        "total": col_total,
    }


modelo, colnames = train_model()

# ---------- 2. Interfaz de usuario ----------

st.title("Predicción de Abandono de Clientes - Telco")
st.write("Ingrese los datos del cliente para estimar la probabilidad de abandono.")

# Entradas del usuario (labels en español)

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

# Construir el dataframe del cliente usando LOS NOMBRES REALES del CSV
datos_dict = {}

if colnames["contract"] is not None:
    datos_dict[colnames["contract"]] = Contrato
if colnames["internet"] is not None:
    datos_dict[colnames["internet"]] = ServicioInternet
if colnames["payment"] is not None:
    datos_dict[colnames["payment"]] = MetodoPago
if colnames["tenure"] is not None:
    datos_dict[colnames["tenure"]] = Permanencia
if colnames["monthly"] is not None:
    datos_dict[colnames["monthly"]] = CargosMensuales
if colnames["total"] is not None:
    datos_dict[colnames["total"]] = CargosTotales

datos_cliente = pd.DataFrame([datos_dict])

if st.button("Predecir abandono"):
    prob = modelo.predict_proba(datos_cliente)[0][1]

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
