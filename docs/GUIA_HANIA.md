# Rol 3 — ANN/MLP e inferencia espacial

**Proyecto:** A Comparative Analysis of Machine Learning Architectures for Satellite-Based Land Cover Classification

**Responsable de este componente:** Hania

**Corte de resultados:** 17 de noviembre de 2026

**Rama de desarrollo:** `feature/hania-ann-inference`

## 1. Alcance

Este componente implementa una red neuronal multicapa (ANN/MLP), la evalúa mediante las particiones espaciales oficiales del equipo y realiza inferencia sobre la escena Sentinel-2. La comparación general del proyecto considera cinco modelos: Regresión Logística (baseline), Decision Tree, SVM (RBF), KNN y ANN/MLP.

El área de estudio definida por el equipo es San José del Palmar, Chocó (Colombia), para una escena Sentinel-2 del 18 de agosto de 2026. El stack entregado tiene diez bandas en este orden exacto: `B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`.

## 2. Datos y validación

- Dataset tabular: `data/training/dataset.tsv` (14.155 filas, cinco clases). Incluye `Latitude`, `Longitude`, `Polygon_ID`, `Class_ID`, diez bandas y `Class`.
- Particiones oficiales: `data/training/spatial_partitions.json`.
- Desarrollo: 11.447 muestras; prueba independiente: 2.708 muestras; validación cruzada: cuatro folds espaciales por `Polygon_ID`.
- Predictores: únicamente las diez bandas. `Latitude`, `Longitude`, `Polygon_ID` y `Class_ID` no se usan como variables predictoras.
- La selección de hiperparámetros se hace únicamente sobre desarrollo. El test independiente se utiliza después de seleccionar la configuración.

## 3. ANN implementada

**Archivo:** `src/model_ann.py`.

Se utiliza `Pipeline(StandardScaler(), MLPClassifier())` con `GridSearchCV`, `scoring="f1_macro"`, semilla 42 y los cuatro folds oficiales. La búsqueda evalúa tres arquitecturas (`[64]`, `[128, 64]`, `[64, 32]`) y dos valores de `alpha` (`0.0001`, `0.001`) con `max_iter=500`: seis combinaciones y 24 ajustes de validación cruzada, más el ajuste final.

**Configuración seleccionada:** dos capas ocultas de 64 y 32 neuronas; `alpha=0.0001`; `max_iter=500`.

| Métrica de ANN | Resultado |
|---|---:|
| CV Macro-F1 (media ± desviación) | 0.7103 ± 0.0866 |
| Test Macro-F1 | 0.7075 |
| Test Accuracy | 88.74 % |
| Test Kappa de Cohen | 0.8146 |
| Test Weighted-F1 | 0.8929 |
| Tiempo de entrenamiento registrado | 103.64 s |

**F1 por clase en el test:** Bosque 0.9996; Suelo Desnudo y Roca 0.8885; Agua 0.6479; Urbano y Vías 0.5926; Pastos y Cultivos 0.4087. La interpretación debe considerar el desbalance de clases y el reducido número de polígonos independientes en algunas coberturas. Un buen resultado sobre píxeles del mismo polígono no equivale a validación de campo independiente.

**Artefactos de ANN:**

- `models/ann.joblib` (Pipeline completo serializado).
- `models/ann_provenance.json` (huellas SHA-256 y procedencia de las particiones).
- `outputs/metrics/ann.json`.
- `outputs/figures/ann_confusion_matrix.png`.
- `outputs/metrics/comparison_table.md` y `.tex` (tabla de las cinco arquitecturas).

La ANN alcanzó el mayor Macro-F1 de los cinco modelos registrados, pero SVM presentó mayor Accuracy y Kappa. La decisión del modelo final para la entrega requiere una justificación multicriterio.

## 4. Inferencia espacial

**Archivo:** `src/predict_raster_hania.py`. Es un script adicional que no modifica `src/predict_raster.py` de Nicolás.

### Entradas locales (fuera del repositorio Git)

- `data/processed/stack_10bands.tif`.
- `data/processed/mask_valid.tif`.

El stack reporta 1054 × 681 píxeles, resolución de 10 m, CRS EPSG:9377 y descripción de las diez bandas en el orden esperado. La máscara tiene la misma geometría espacial.

### Regla aplicada de validez

Se clasifica un píxel solo si la máscara oficial es positiva, los valores de las diez bandas son finitos y ninguna banda contiene el valor NoData declarado por el stack (`0.0`). En esta escena:

| Concepto | Píxeles |
|---|---:|
| Válidos según máscara oficial | 471.579 |
| Excluidos adicionalmente por NoData en alguna banda | 129.054 |
| Clasificados con diez bandas válidas | **342.525** |

Los ceros se concentran sobre todo en `B05`, `B06`, `B07` y `B8A`. Rasterio informa `MaskFlags.nodata`, que proviene del valor NoData declarado; esto **no constituye una validación externa independiente**. El origen y significado de los ceros deben confirmarse con el responsable del preprocesamiento.

### Salidas

Para cada modelo se genera `outputs/rasters/landcover_<modelo>.tif`, con una banda categórica, códigos 1–5, NoData=0 y la georreferenciación de la escena. Estos TIFF no se incluyen en Git debido a `.gitignore`.

Las estadísticas del filtrado de diez bandas **se escriben en**:

`outputs/metrics/inferencia_10bandas/area_stats_<modelo>.json`

Esta ubicación evita sobrescribir `outputs/metrics/area_stats_logistic_regression.json` y `outputs/metrics/area_stats_svm.json`, que contienen resultados históricos del equipo bajo otros criterios de inferencia.

**Superficies ANN (hectáreas, resultado preliminar):** Bosque 2.189,51; Suelo Desnudo y Roca 1.172,13; Pastos y Cultivos 38,22; Urbano y Vías 19,93; Agua 5,46. Total: **3.425,25 ha** (342.525 píxeles × 100 m²).

### Advertencia de compatibilidad

El archivo de Regresión Logística produjo `InconsistentVersionWarning` al cargarse: serializado con scikit-learn 1.9.1 y utilizado aquí con 1.5.2. El mapa inferido con ese objeto debe considerarse provisional hasta comprobar la compatibilidad o regenerar el baseline en el entorno acordado. ANN, Decision Tree, SVM y KNN se cargaron sin esa advertencia.

## 5. Ejecución reproducible

Desde la raíz del repositorio con Python 3.12 y las dependencias del `requirements.txt`:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -v
python src/model_ann.py --check-only
python src/predict_raster_hania.py --check-inputs
```

Para **reentrenar** la ANN (solo si se necesita reproducir los artefactos):

```bash
python src/model_ann.py --jobs 2
```

Para inferir la ANN, sin volver a entrenarla:

```bash
python src/predict_raster_hania.py --model ann
```

Para inferir los cinco modelos (teniendo en cuenta la advertencia de compatibilidad de LR):

```bash
python src/predict_raster_hania.py --model all
```

En la sesión del 17/11/2026 las ocho pruebas automatizadas terminaron correctamente. También se comprobó que los cinco TIFF generados tenían una única banda categórica, EPSG:9377, NoData 0, los códigos esperados y recuentos de 342.525 píxeles coherentes con sus estadísticas.

## 6. Evidencias cartográficas

- `outputs/figures/Mapa de cobertura ANN.pdf` — diseño exportado desde QGIS.
- `outputs/figures/Mapa de cobertura terrestre - ANN.png` — imagen exportada desde QGIS.
- `outputs/rasters/Clasificacion_Cobertura_ANN.qgz` — proyecto editable de QGIS.
- `outputs/rasters/landcover_ann.tif` — GeoTIFF categórico local (ignorado por Git).

La composición incluye título, leyenda de las cinco clases, norte, escala, cuadro de superficies y nota técnica. El proyecto `.qgz` depende del GeoTIFF y puede necesitar que se actualice la ruta de la capa al abrirse en otro equipo. Si alguien clona el repositorio sin los TIFF de entrada/salida, debe obtener el stack y la máscara compartidos por el equipo y ejecutar de nuevo la inferencia. El PDF/PNG pueden visualizarse sin esos rásteres.

## 7. Pendientes para cierre y defensa

1. Confirmar con Nicolás la generación de `mask_valid.tif` y las bandas con cero/NoData. Mantener explícita la diferencia entre máscara oficial y píxeles clasificables.
2. Alinear versiones de scikit-learn o regenerar el baseline de Regresión Logística antes de declarar definitivo su mapa.
3. Elegir y justificar el modelo final utilizando conjuntamente Macro-F1, Accuracy, Kappa, métricas por clase y limitaciones del test espacial.
4. Asegurar la entrega del GeoTIFF final, del código en ZIP y del informe técnico, además de la visualización QGIS y las estadísticas de área.
5. Incorporar el resultado al informe técnico del equipo y documentar las conclusiones sin presentar las predicciones como verdad de terreno independiente.

**No modificar archivos originales de otros integrantes al integrar este componente.** Mantener los rásteres de entrada y salida pesados fuera de Git; compartirlos por la vía acordada con el equipo.
