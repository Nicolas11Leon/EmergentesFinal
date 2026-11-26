# EmergentesFinal

Clasificación supervisada de coberturas terrestres con Sentinel-2 en San José del Palmar (Chocó, Colombia).
Proyecto final de Tecnologías Emergentes, Pontificia Universidad Javeriana.

- **Escena:** Sentinel-2 L2A del 18/08/2026, 10 bandas (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
- **Grilla:** 1054 × 681 píxeles, 10 m, EPSG:9377 (MAGNA-SIRGAS Origen Nacional).
- **Clases:** 1 Bosque · 2 Pastos y Cultivos · 3 Suelo Desnudo y Roca · 4 Urbano y Vías · 5 Agua (0 = NoData).
- **Modelos:** Regresión Logística (baseline), Árbol de Decisión, SVM (RBF), KNN y ANN/MLP.
- **Validación:** partición espacial por `Polygon_ID` (116 polígonos de desarrollo con 4 folds, 29 de prueba).

## Estructura

| Ruta | Contenido |
|---|---|
| `config/` | Parámetros (`project.yaml`), clases y colores (`classes.json`) |
| `data/vectors/` | ROI (`roi_palmar.geojson`), polígonos de verdad de terreno (`ground_truth.gpkg`) |
| `data/training/` | `dataset.tsv` (14.155 píxeles, 145 polígonos) y `spatial_partitions.json` |
| `src/` | Preparación, extracción TSV, partición, modelos, inferencia y auditoría |
| `models/` | Cinco modelos entrenados (`.joblib`) y su procedencia |
| `outputs/metrics/` | Métricas por modelo, tabla comparativa y áreas (`inferencia_10bandas/`) |
| `outputs/figures/` | Matrices de confusión, importancia de bandas y mapa temático ANN |
| `outputs/rasters/` | Simbología QGIS (`landcover_style.qml`), proyecto QGIS y GeoTIFF de salida |
| `docs/` | Informe técnico, decisiones de inferencia y material de sustentación |
| `tests/` | Pruebas automáticas (`pytest`) |

## Entorno (probado en Windows)

Python 3.12.10:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
```

## Entradas externas (no versionadas)

Los GeoTIFF no se suben a Git (`.gitignore`). La inferencia necesita el ZIP oficial
`Emergentes_Rasters_Inferencia.zip`, con `stack_10bands.tif` y `mask_valid.tif`. Ambos
archivos se copian en `data/processed/`.

## Ejecución oficial (PowerShell, desde la raíz del repositorio)

```powershell
# 1. Auditar los insumos raster (stack de 10 bandas y máscara)
.\.venv\Scripts\python.exe .\src\audit_rasters_final.py --zip "RUTA\Emergentes_Rasters_Inferencia.zip" --source-only

# 2. Inferencia oficial de los cinco modelos (genera outputs/rasters/landcover_<modelo>.tif
#    y outputs/metrics/inferencia_10bandas/area_stats_<modelo>.json)
.\.venv\Scripts\python.exe .\src\predict_raster_hania.py --model all

# 3. Auditoría píxel a píxel de los cinco GeoTIFF contra la máscara y los JSON
.\.venv\Scripts\python.exe .\src\audit_rasters_final.py --zip "RUTA\Emergentes_Rasters_Inferencia.zip" `
  --rasters-dir .\outputs\rasters --stats-dir .\outputs\metrics\inferencia_10bandas `
  --report .\outputs\auditorias\auditoria_final_5_modelos.json
```

Las etapas previas (stack, extracción del TSV, partición y entrenamiento) están en
`src/01_prepare_stack.py`, `src/02_extract_tsv.py`, `src/spatial_split.py`,
`src/model_logistic_regression.py`, `src/model_classical.py` y `src/model_ann.py`.
Los modelos ya entrenados se reutilizan: sus métricas de prueba se reproducen exactamente
desde los `.joblib`.

**Notas:**
- `src/predict_raster.py` es la versión anterior de la inferencia: aplica la máscara y `B02 > 0`,
  pero no el criterio de diez bandas. No debe usarse para resultados oficiales
  (ver `docs/DECISIONES_INFERENCIA.md`).
- Sus salidas históricas están en `outputs/metrics/legacy_predict_raster/` y no son resultados oficiales.
- `models/logistic_regression.joblib` fue serializado con scikit-learn 1.9.1. Con la versión 1.5.2
  aparece `InconsistentVersionWarning`, pero las métricas de prueba se reproducen exactamente.

## Resultados (conjunto de prueba independiente, 2.708 píxeles)

| Modelo | Overall Accuracy | Kappa | Macro-F1 |
|---|---:|---:|---:|
| ANN / MLP (64, 32) | 88,74 % | 0,8146 | **0,7075** |
| SVM (RBF, C=10) | **89,70 %** | **0,8283** | 0,6889 |
| Árbol de Decisión (d=8) | 86,41 % | 0,7827 | 0,6794 |
| Regresión Logística | 82,87 % | 0,7343 | 0,6516 |
| KNN (k=15) | 87,70 % | 0,7936 | 0,6159 |

El mapa principal usa ANN porque tiene el mayor Macro-F1, la métrica priorizada para equilibrar
las cinco clases. SVM tiene mayor Accuracy y Kappa. No se hicieron pruebas de significancia
estadística entre modelos.

## Equipo

Nicolás León · Andrés Felipe Barrios · Hania Valentina Carreño · Jorge Sierra
