"""
Entrenamiento de un Random Forest para diagnosticar el estado de un motor
(sano / desbalanceo / holgura / sobrecarga) a partir de 3 sensores:
vibración, corriente y temperatura.

Uso:
    python train.py

Lee:
    data/raw/vib_<estado>_<fecha>.csv  -> rms, pico, crest, fdom, adom
    data/raw/cur_<estado>_<fecha>.csv  -> I_mA, V_V, P_mW
    data/raw/tmp_<estado>_<fecha>.csv  -> T_C

Genera:
    data/sensores.csv  -> dataset unificado (una fila por lectura de vibración)
    modelo.joblib      -> modelo entrenado + métricas para el dashboard
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
DATA_PATH = Path("data/sensores.csv")
MODEL_PATH = Path("modelo.joblib")

FEATURES = ["rms", "pico", "crest", "fdom", "adom", "I_mA", "V_V", "P_mW"]
# T_C (temperatura) se guarda para el dashboard pero NO entra al modelo:
# cada sesión se grabó a una temperatura distinta y sin solapes, así que el
# modelo aprendería a reconocer la sesión en vez de la avería (data leakage).
EXTRA = ["T_C"]
TARGET = "label"
CLASES = ["sano", "desbalanceo", "holgura", "sobrecarga"]

# Máxima distancia temporal (ms) para emparejar lecturas de distintos sensores
TOLERANCIA_MS = 1000


# ---------------------------------------------------------------------------
# 1) Unir los 3 sensores de cada sesión por marca de tiempo
# ---------------------------------------------------------------------------
def cargar_sesion(vib_path: Path) -> pd.DataFrame:
    sufijo = vib_path.name.removeprefix("vib_")  # p.ej. "sano_20260606_201014.csv"

    vib = pd.read_csv(vib_path).drop(columns=["speed"])
    cur = pd.read_csv(RAW_DIR / f"cur_{sufijo}").drop(columns=["speed", "label"])
    tmp = pd.read_csv(RAW_DIR / f"tmp_{sufijo}").drop(columns=["speed", "label"])

    # La vibración se mide ~2.5 veces por segundo y corriente/temperatura 1 vez,
    # así que a cada lectura de vibración le asignamos la lectura más cercana.
    df = pd.merge_asof(vib, cur, on="ts_ms", direction="nearest", tolerance=TOLERANCIA_MS)
    df = pd.merge_asof(df, tmp, on="ts_ms", direction="nearest", tolerance=TOLERANCIA_MS)

    df["t_s"] = (df["ts_ms"] - df["ts_ms"].iloc[0]) / 1000  # segundos desde el inicio
    return df.dropna()


def cargar_datos() -> pd.DataFrame:
    sesiones = [cargar_sesion(p) for p in sorted(RAW_DIR.glob("vib_*.csv"))]
    if not sesiones:
        raise SystemExit(f"No hay ficheros vib_*.csv en {RAW_DIR}")
    df = pd.concat(sesiones, ignore_index=True)
    return df[["t_s", *FEATURES, *EXTRA, TARGET]]


def main():
    df = cargar_datos()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset: {len(df)} filas -> {DATA_PATH}")
    print(df[TARGET].value_counts().to_string())

    # -----------------------------------------------------------------------
    # 2) Train / test split (estratificado para mantener la proporción de clases)
    # -----------------------------------------------------------------------
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------------------------------------------------
    # 3) Random Forest (mismos hiperparámetros que en la teoría)
    # -----------------------------------------------------------------------
    bosque = RandomForestClassifier(
        n_estimators=100,          # Cantidad de árboles
        max_features="sqrt",       # Variables por división
        max_depth=10,              # Profundidad máxima
        min_samples_split=5,       # Mínimo para dividir
        class_weight="balanced",   # Compensa que haya menos datos de holgura
        oob_score=True,            # Validación Out-Of-Bag automática
        random_state=42,
        n_jobs=-1,
    )
    bosque.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # 4) Evaluación
    # -----------------------------------------------------------------------
    y_pred = bosque.predict(X_test)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "oob": bosque.oob_score_,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    importancias = pd.Series(bosque.feature_importances_, index=FEATURES).sort_values(ascending=False)

    print(f"\nAccuracy test: {metricas['accuracy']:.3f}")
    print(f"F1 macro:      {metricas['f1']:.3f}")
    print(f"OOB score:     {metricas['oob']:.3f}")
    print("\nImportancia de variables:")
    print(importancias.round(3).to_string())

    # -----------------------------------------------------------------------
    # 5) Guardar modelo + info para el dashboard
    # -----------------------------------------------------------------------
    joblib.dump(
        {
            "modelo": bosque,
            "features": FEATURES,
            "clases": CLASES,
            "metricas": metricas,
            "importancias": importancias.to_dict(),
            "matriz_confusion": confusion_matrix(y_test, y_pred, labels=CLASES).tolist(),
        },
        MODEL_PATH,
        compress=3,
    )
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
