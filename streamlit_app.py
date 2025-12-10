import streamlit as st
import pandas as pd
import joblib

st.title("Predicción de Abandono de Clientes - Telco")
st.write("Ingrese los datos del cliente para estimar la probabilidad de abandono.")

# Cargar modelo y procesador
modelo = joblib.load("modelo_telco.pkl")
preprocesador = joblib.load("preprocesador.pkl")  # OneHotEncoder + scaler (si lo usaste)


# --- ENTRADAS DEL USUARIO ---
contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Método de pago", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

tenure = st.slider("Permanencia (meses)", 0, 72, 10)
monthly = st.number_input("Cargos mensuales", min_value=0.0, max_value=200.0, value=80.0)
total = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, value=1000.0)


# Crear dataframe
datos_cliente = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total],
    "Contract": [contract],
    "InternetService": [internet],
    "PaymentMethod": [payment]
})


# --- PREDICCIÓN ---
if st.button("Predecir abandono"):

    # Preprocesar los datos del cliente igual que el modelo
    datos_procesados = preprocesador.transform(datos_cliente)

    # Obtener las probabilidades
    probs = modelo.predict_proba(datos_procesados)
    classes = list(modelo.classes_)

    # Manejo correcto de índice de clase 1
    if 1 in classes:
        idx = classes.index(1)
        prob = float(probs[0][idx])
    else:
        prob = 0.0

    st.subheader(f"Probabilidad de abandono: {prob:.2%}")

    # --- Indicador visual ---
    st.progress(prob)

    # --- Mensaje según riesgo ---
    if prob >= 0.70:
        st.error("🔴 Riesgo ALTO de abandono")
        st.write("👉 Recomendación: ofrecer descuento relevante, migrar a contrato anual o contactar al cliente.")
    elif prob >= 0.40:
        st.warning("🟠 Riesgo MEDIO de abandono")
        st.write("👉 Recomendación: enviar promociones personalizadas y mejorar la comunicación.")
    else:
        st.success("🟢 Riesgo BAJO de abandono")
        st.write("👉 Recomendación: mantener la estrategia actual y programas de fidelización.")
