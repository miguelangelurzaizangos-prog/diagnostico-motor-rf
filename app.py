"""
Dashboard de sensores de un motor + diagnóstico de averías con Random Forest.

Uso:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/sensores.csv")
MODEL_PATH = Path("modelo.joblib")

DESCRIPCION = {
    "rms": "Vibración RMS (g)",
    "pico": "Vibración pico (g)",
    "crest": "Factor de cresta",
    "fdom": "Frecuencia dominante (Hz)",
    "adom": "Amplitud dominante",
    "I_mA": "Corriente (mA)",
    "V_V": "Tensión (V)",
    "P_mW": "Potencia (mW)",
    "T_C": "Temperatura (ºC)",
}
ICONO = {"sano": "✅", "desbalanceo": "⚖️", "holgura": "🔩", "sobrecarga": "🔥"}

st.set_page_config(page_title="Diagnóstico de motor", page_icon="⚙️", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos y modelo (se cachean para no recargar en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def cargar_modelo() -> dict:
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Faltan `data/sensores.csv` o `modelo.joblib`. Ejecuta primero: `python train.py`")
    st.stop()

df = cargar_datos()
artefacto = cargar_modelo()
modelo = artefacto["modelo"]
features = artefacto["features"]
clases = artefacto["clases"]
metricas = artefacto["metricas"]


# ---------------------------------------------------------------------------
# Barra lateral: filtros
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
sel_estados = st.sidebar.multiselect(
    "Estados del motor", clases, default=clases, format_func=lambda c: f"{ICONO[c]} {c}"
)
datos = df[df["label"].isin(sel_estados)]

if datos.empty:
    st.warning("Selecciona al menos un estado.")
    st.stop()


# ---------------------------------------------------------------------------
# Cabecera + KPIs
# ---------------------------------------------------------------------------
st.title("⚙️ Diagnóstico de motor con Random Forest")
st.caption("Lecturas reales de vibración, corriente y temperatura en 4 estados de funcionamiento.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lecturas", f"{len(datos):,}".replace(",", "."))
k2.metric("Estados", datos["label"].nunique())
k3.metric("Vibración RMS media", f"{datos['rms'].mean():.3f} g")
k4.metric("Temperatura media", f"{datos['T_C'].mean():.1f} ºC")

tab_datos, tab_modelo, tab_pred = st.tabs(["📈 Datos", "🌲 Modelo", "🔮 Diagnóstico"])


# ---------------------------------------------------------------------------
# Pestaña 1: exploración de datos
# ---------------------------------------------------------------------------
with tab_datos:
    sensor = st.selectbox("Sensor", [*features, "T_C"], format_func=lambda s: DESCRIPCION[s])

    st.subheader(f"{DESCRIPCION[sensor]} a lo largo de cada sesión")
    serie = datos.pivot_table(index="t_s", columns="label", values=sensor)
    st.line_chart(serie)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lecturas por estado")
        st.bar_chart(datos["label"].value_counts())
    with c2:
        st.subheader("Valor medio de cada sensor por estado")
        st.dataframe(
            datos.groupby("label")[[*features, "T_C"]].mean().T.style.format("{:.3f}"),
            width="stretch",
        )

    with st.expander("Ver datos en bruto"):
        st.dataframe(datos, width="stretch")


# ---------------------------------------------------------------------------
# Pestaña 2: rendimiento del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy (test)", f"{metricas['accuracy']:.1%}")
    m2.metric("F1 macro", f"{metricas['f1']:.2f}")
    m3.metric("OOB score", f"{metricas['oob']:.1%}")
    st.caption(
        f"Entrenado con {metricas['n_train']} lecturas y evaluado con {metricas['n_test']}. "
        f"{modelo.n_estimators} árboles, max_depth={modelo.max_depth}, max_features='{modelo.max_features}'."
    )
    st.info(
        "La temperatura **no** se usa en el modelo: cada sesión se grabó a una temperatura distinta "
        "y el bosque aprendía a reconocer la sesión en lugar de la avería (data leakage)."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Importancia de variables")
        imp = pd.Series(artefacto["importancias"]).rename(index=DESCRIPCION)
        st.bar_chart(imp.sort_values(ascending=False))
    with c2:
        st.subheader("Matriz de confusión")
        cm = pd.DataFrame(
            artefacto["matriz_confusion"],
            index=[f"Real: {c}" for c in clases],
            columns=[f"Pred: {c}" for c in clases],
        )
        st.dataframe(cm, width="stretch")


# ---------------------------------------------------------------------------
# Pestaña 3: diagnóstico interactivo
# ---------------------------------------------------------------------------
with tab_pred:
    st.write("Elige una lectura real como punto de partida y ajusta los sliders para ver qué diagnostica el bosque.")

    base = st.selectbox("Partir de una lectura típica de…", clases, format_func=lambda c: f"{ICONO[c]} {c}")
    medianas = df[df["label"] == base][features].median()

    entrada = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        lo, hi = float(df[f].min()), float(df[f].max())
        if lo == hi:
            hi = lo + 1
        entrada[f] = cols[i % 3].slider(DESCRIPCION[f], lo, hi, float(medianas[f]), key=f"{base}_{f}")

    X = pd.DataFrame([entrada])[features]
    probs = pd.Series(modelo.predict_proba(X)[0], index=modelo.classes_).reindex(clases)
    pred = probs.idxmax()

    # Voto de cada árbol (para enseñar el "voto mayoritario" del bosque)
    votos = pd.Series(
        [modelo.classes_[int(arbol.predict(X.values)[0])] for arbol in modelo.estimators_]
    ).value_counts().reindex(clases, fill_value=0)

    st.divider()
    if pred == "sano":
        st.success(f"{ICONO[pred]} Diagnóstico: **{pred.upper()}** · {probs[pred]:.1%} de confianza")
    else:
        st.error(f"{ICONO[pred]} Diagnóstico: **{pred.upper()}** · {probs[pred]:.1%} de confianza")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probabilidad por estado")
        st.bar_chart(probs)
    with c2:
        st.subheader(f"Votos de los {len(modelo.estimators_)} árboles")
        st.bar_chart(votos)
