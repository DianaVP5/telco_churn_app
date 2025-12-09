import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------- 1. Entrenar el modelo al iniciar la app ----------

@st.cache_resource
def train_model():
    # Cargar datos (usa el nombre exacto de tu CSV en el repo)
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # --- Mostrar columnas disponibles (útil para debug si algo falla) ---
    # st.write("Columnas del CSV:", list(df.columns))

    # ---------- Detectar nombres reales de columnas ----------

    # Columna objetivo: Churn / Abandono
    if "Churn" in df.columns:
        col_target = "Churn"
    elif "Abandono" in df.columns:
        col_target = "Abandono"
    else:
        raise ValueError(
            f"No se encontró la columna de abandono ('Churn' o 'Abandono'). "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Contrato: Contract / Contrato
    if "Contract" in df.columns:
        col_contract = "Contract"
    elif "Contrato" in df.columns:
        col_contract = "Contrato"
    else:
        col_contract = None

    # Servicio de Internet: InternetService / ServicioInternet
    if "InternetService" in df.columns:
        col_internet = "InternetService"
    elif "ServicioInternet" in df.columns:
        col_internet = "ServicioInternet"
    else:
        col_internet = None

    # Método de pago: PaymentMethod / MetodoPago
    if "PaymentMethod" in df.columns:
        col_payment = "PaymentMethod"
    elif "MetodoPago" in df.columns:
        col_payment = "MetodoPago"
    else:
        col_payment = None

    # Permanencia: tenure / Permanencia
    if "tenure" in df.columns:
        col_tenure = "tenure"
    elif "Permanencia" in df.columns:
        col_tenure = "Permanencia"
    else:
        col_tenure = None

    # Cargos mensuales: MonthlyCharges / CargosMensuales
    if "MonthlyCharges" in df.columns:
        col_monthly = "MonthlyCharges"
    elif "CargosMensuales" in df.columns:
        col_monthly = "CargosMensuales"
    else:
        col_monthly = None

    # Cargos totales: TotalCharges / CargosTotales (puede no existir)
    if "TotalCharges" in df.columns:
        col_total = "TotalCharges"
    elif "CargosTotales" in df.columns:
        col_total = "CargosTotales"
    else:
        col_total = None  # la hacemos opcional

    # ---------- Crear variable objetivo numérica ----------

    df["abandono_flag"] = df[col_target].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "si", "1", "true"] else 0
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
        # asegurar tipo numérico
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
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(drop="first"), categoricas),
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
