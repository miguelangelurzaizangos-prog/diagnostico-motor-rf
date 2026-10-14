# Diagnóstico de motor con Random Forest + Streamlit

Dashboard que muestra lecturas reales de **vibración, corriente y temperatura** de un motor
y usa un **Random Forest** para diagnosticar su estado:

| Estado | Descripción |
|---|---|
| ✅ sano | Funcionamiento normal |
| ⚖️ desbalanceo | Masa descompensada en el eje |
| 🔩 holgura | Piezas flojas / juego mecánico |
| 🔥 sobrecarga | Motor trabajando por encima de su carga |

## Estructura

```text
sensores-rf-app/
├── app.py              # Dashboard Streamlit
├── train.py            # Une los sensores y entrena el Random Forest
├── modelo.joblib       # Modelo entrenado
├── requirements.txt
├── data/
│   ├── raw/            # CSV originales: vib_*, cur_*, tmp_*
│   └── sensores.csv    # Dataset unificado (lo genera train.py)
└── README.md
```

## Datos

Cada sesión de grabación tiene 3 ficheros (uno por sensor):

- `vib_<estado>_<fecha>.csv` → `rms`, `pico`, `crest`, `fdom`, `adom` (~2.5 lecturas/s)
- `cur_<estado>_<fecha>.csv` → `I_mA`, `V_V`, `P_mW` (1 lectura/s)
- `tmp_<estado>_<fecha>.csv` → `T_C` (1 lectura/s)

`train.py` los une por marca de tiempo (`ts_ms`) con `pd.merge_asof`: a cada lectura de
vibración se le asigna la lectura de corriente y temperatura más cercana.

## Modelo

Random Forest con los hiperparámetros vistos en clase (`n_estimators=100`,
`max_features='sqrt'`, `max_depth=10`, `min_samples_split=5`, `oob_score=True`).

| Métrica | Valor |
|---|---|
| Accuracy (test) | 98.4 % |
| F1 macro | 0.98 |
| OOB score | 99.3 % |

**Ojo con la temperatura:** la primera versión daba un 100 % de acierto y la variable
más importante era `T_C`. Cada sesión se grabó a una temperatura distinta (sin solaparse),
así que el modelo reconocía *la sesión* y no *la avería* → **data leakage**. Por eso la
temperatura se muestra en el dashboard pero no entra al modelo.

## Ejecutar en local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py          # opcional: el repo ya incluye modelo.joblib
streamlit run app.py
```

## Deploy en Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
