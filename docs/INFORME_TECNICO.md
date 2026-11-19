# INFORME TÉCNICO FINAL · FASES 1, 2 Y 3
## Análisis Comparativo de Arquitecturas de Machine Learning para la Clasificación de Coberturas Terrestres con Imágenes Sentinel-2 en San José del Palmar, Chocó

**PROYECTO FINAL · TECNOLOGÍAS EMERGENTES**

### INTEGRANTES
- **Nicolás León**
- **Andrés Felipe Barrios**
- **Hania Valentina Carreño**
- **Jorge Sierra**

**Profesor:** Andrés Oswaldo Calderón Romero, Ph.D.  
**Institución:** Pontificia Universidad Javeriana · Bogotá, Colombia  
**Fecha de elaboración:** 19 de noviembre de 2026  

**ALCANCE**  
Informe técnico final que integra las tres fases del proyecto: (1) definición del caso de estudio, delimitación espacial y extracción reproducible del conjunto tabular `dataset.tsv`; (2) validación espacial sin fuga de polígonos, optimización de hiperparámetros (`GridSearchCV` con `Macro-F1`) y evaluación comparativa de las cinco arquitecturas de aprendizaje automático con sus matrices de confusión, exactitud global, índice Kappa, precisión, exhaustividad (*recall*), F1 por clase e importancia espectral de bandas; y (3) inferencia ráster sobre la escena completa en `EPSG:9377`, exportación de GeoTIFFs categóricos verificados, visualización cartográfica en QGIS y cuantificación espacial de áreas en hectáreas ($\text{ha}$) y kilómetros cuadrados ($\text{km}^2$).

---

## 1. Resumen ejecutivo

El equipo construyó un conjunto tabular de **14.155 píxeles etiquetados** a partir de diez bandas espectrales Sentinel-2 Nivel-2A. Se conservaron **145 polígonos efectivos** con muestras válidas y se separaron **11.447 observaciones para desarrollo** (116 polígonos) y **2.708 para prueba independiente** (29 polígonos), sin compartir ningún `Polygon_ID` entre ambos conjuntos. Sobre cuatro *folds* espaciales de desarrollo se entrenaron y optimizaron cinco arquitecturas de clasificación supervisada —**Regresión Logística (baseline)**, **Árbol de Decisión (CART)**, **Support Vector Machines (SVM RBF)**, **K-Nearest Neighbors (KNN)** y **Red Neuronal Artificial (ANN/MLP)**— mediante `GridSearchCV`, utilizando `Macro-F1` como métrica objetiva de selección.

En la partición de prueba reservada ($n = 2.708$), **SVM RBF** alcanzó la mayor Exactitud Global (**89,70 %**, 2.429 aciertos) y el mayor índice Kappa (**0,8283**), mientras que **ANN/MLP** obtuvo el mayor **Macro-F1 en prueba (0,7075)** y el mayor **Macro-F1 promedio en validación cruzada de desarrollo (0,7103 ± 0,0866)**. Siguiendo el protocolo de selección sin optimización contra el conjunto de prueba, **ANN/MLP** fue desplegada como arquitectura principal para la inferencia espacial sobre los **342.525 píxeles válidos con cobertura completa en las diez bandas (3.425,25 ha = 34,2525 $\text{km}^2$)**, contrastando sus estimaciones de superficie frente a los demás modelos y verificando la coincidencia exacta pixel a pixel del archivo `outputs/rasters/landcover_ann.tif` con la cartografía temática elaborada en QGIS.

| Indicador del proyecto | Valor confirmado y verificado |
| :--- | :--- |
| **Dataset tabular (`dataset.tsv`)** | 14.155 filas · 10 bandas espectrales · 5 clases · 15 columnas |
| **Separación espacial (`Polygon_ID`)** | 116 polígonos dev (11.447 filas) · 29 polígonos test (2.708 filas) |
| **Validación y búsqueda (HPO)** | 4 *folds* espaciales · `GridSearchCV` · `scoring="f1_macro"` |
| **Arquitecturas comparadas** | Regresión Logística, Árbol de Decisión, SVM RBF, KNN y ANN/MLP |
| **Mejor Overall Accuracy y Kappa (test)** | SVM RBF: **89,70 %** (2.429/2.708) · Kappa: **0,8283** |
| **Mejor Macro-F1 en CV y en test** | ANN/MLP: **0,7103 ± 0,0866** (CV) · **0,7075** (test) |
| **Bandas más influyentes (Árbol CART)** | `B08` NIR (33,90 %), `B04` Rojo (32,58 %), `B02` Azul (24,18 %) y `B11` SWIR (4,42 %) |
| **Dominio de inferencia ráster verificado** | 342.525 píxeles válidos en 10 bandas · **3.425,25 ha** (**34,2525 $\text{km}^2$**) · `EPSG:9377` |

---

## 2. Contexto, región de interés (ROI) y objetivo (Fase 1)

El Servicio Geológico Colombiano reportó un sismo de magnitud $M_w$ 7,4 el 10 de agosto de 2026 a las 7:34 a. m. (12:34 UTC), con epicentro en jurisdicción de San José del Palmar (Chocó) y profundidad aproximada de 103 km [1]. Como contexto de monitoreo ambiental y territorial en la vertiente occidental de la Cordillera Occidental (cañón del río Ingará), el equipo seleccionó la región de estudio analizada previamente mediante detección de cambios y trabajó sobre una única escena post-evento **Sentinel-2 MSI Nivel-2A** adquirida el **18 de agosto de 2026**, ocho días después del movimiento telúrico.

**Objetivo general:** Implementar un flujo de trabajo reproducible de Inteligencia Artificial Geoespacial (GeoAI), independiente del software SIG de escritorio para la fase de modelado, que transforme las bandas multiespectrales de Sentinel-2 en un conjunto de entrenamiento tabular (`.tsv`), evalúe comparativamente cinco arquitecturas de aprendizaje automático bajo optimización de hiperparámetros y validación espacial estricta, y despliegue el modelo óptimo sobre el stack ráster completo para producir un GeoTIFF categórico georreferenciado, su visualización temática en QGIS y la cuantificación de superficie por cobertura en hectáreas y kilómetros cuadrados.

Las cinco categorías de cobertura terrestre evaluadas son:
1. **Bosque** (`Class_ID = 1`)
2. **Pastos y Cultivos** (`Class_ID = 2`)
3. **Suelo Desnudo y Roca** (`Class_ID = 3`)
4. **Urbano y Vías** (`Class_ID = 4`)
5. **Agua** (`Class_ID = 5`)

La clasificación mono-temporal posterior al sismo describe la distribución espacial de las coberturas observadas el 18 de agosto de 2026; por sí sola caracteriza el estado físico de la superficie (incluyendo laderas desprovistas de vegetación y cicatrices expuestas), sin atribuir causalidad sísmica exclusiva a la totalidad del suelo desnudo sin un contraste multitemporal.

### Delimitación espacial documentada
El área de estudio corresponde al recorte de la escena y la máscara de píxeles válidos empleada en el stack multiespectral (`1.054 × 681` celdas a $10\text{ m} \times 10\text{ m}$). Su huella espacial se observa en la Figura 1 y se representa en el sistema coordenado oficial **MAGNA-SIRGAS Origen Nacional (`EPSG:9377`)**. Para reproducir el límite exacto se conservan los insumos geoespaciales originales y sus metadatos afines.

---

## 3. Preparación del conjunto de datos y extracción tabular

### 3.1 Imagen, bandas espectrales y referencia espacial
La documentación y configuración del proyecto (`config/project.yaml`) identifican la escena `S2C_MSIL2A_20260818T153621_N0512_R068_T18NUL` (reflectancia de superficie *Bottom-of-Atmosphere*, BOA) reproyectada a **MAGNA-SIRGAS Origen Nacional (`EPSG:9377`)**. El stack consolidado (`data/processed/stack_10bands.tif`) trabaja sobre una grilla común de **10 m** y reúne diez bandas espectrales: **`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11` y `B12`**. Las bandas con resolución nativa de 20 m (`B05`, `B06`, `B07`, `B8A`, `B11` y `B12`) fueron remuestreadas a 10 m para alinear la lectura multiespectral por píxel [2].

| Orden | Banda | Región espectral | Longitud de onda central | Resolución nativa | Grilla de trabajo | Función biofísica principal |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `B02` | Azul visible | 490 nm | 10 m | 10 m | Discriminación agua/suelo y dispersión atmosférica residual |
| 2 | `B03` | Verde visible | 560 nm | 10 m | 10 m | Pico de reflectancia visible de la vegetación y turbidez del agua |
| 3 | `B04` | Rojo visible | 665 nm | 10 m | 10 m | Absorción por clorofila; contraste directo bosque vs. suelo expuesto |
| 4 | `B05` | Red Edge 1 | 705 nm | 20 m | 10 m | Inicio de la rampa del borde rojo; sensibilidad a estrés vegetal |
| 5 | `B06` | Red Edge 2 | 740 nm | 20 m | 10 m | Transición del borde rojo; diferenciación entre dosel denso y pasturas |
| 6 | `B07` | Red Edge 3 | 783 nm | 20 m | 10 m | Hombro superior del Red Edge; estructura foliar y biomasa |
| 7 | `B08` | NIR amplio | 842 nm | 10 m | 10 m | Alta reflectancia celular de vegetación sana y fuerte absorción en agua |
| 8 | `B8A` | NIR estrecho | 865 nm | 20 m | 10 m | Meseta infrarroja estrecha menos sensible al vapor de agua |
| 9 | `B11` | SWIR 1 | 1610 nm | 20 m | 10 m | Contenido hídrico foliar/edáfico; separación suelo mineral, roca y urbano |
| 10 | `B12` | SWIR 2 | 2190 nm | 20 m | 10 m | Minerales arcillosos, sustrato rocoso expuesto e infraestructura |

`B02`–`B04` aportan información del espectro visible; `B05`–`B07` y `B8A` describen la región *Red Edge*/NIR estrecho; `B08` corresponde al infrarrojo cercano amplio; y `B11`–`B12` cubren el infrarrojo de onda corta (SWIR). La grilla de 10 m constituye una resolución de muestreo común: permite integrar las diez variables por celda sin implicar que las bandas nativas de 20 m hayan adquirido detalle espacial sub-píxel nuevo.

*(Ver Figura 1: Composición RGB real de Sentinel-2 bandas B04, B03 y B02 del stack de 10 bandas, y Figura 2: Flujo documentado del preprocesamiento, extracción tabular y separación espacial).*

### 3.2 Conversión de ráster a TSV (`src/02_extract_tsv.py`)
El script [`src/02_extract_tsv.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/02_extract_tsv.py) intersecta los polígonos de verdad de terreno (`data/training/ground_truth.gpkg`) con `stack_10bands.tif`, extrae las reflectancias de cada píxel interior y exporta el archivo tabular separado por tabulaciones [`data/training/dataset.tsv`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/dataset.tsv). El algoritmo aplica inicialmente `all_touched=False` (centro geométrico del píxel dentro del polígono) y, únicamente para polígonos estrechos que no capturan centros de celda, reintenta con `all_touched=True`; descarta valores `NaN`, infinitos y reflectancias fuera del rango físico $[0, 1]$ [2].

En cumplimiento estricto de la Sección 3.1 del enunciado, el archivo `dataset.tsv` contiene **14.155 filas** y **15 columnas**:
- **Coordenadas espaciales:** `Latitude` y `Longitude` (centroide del píxel transformado a WGS84 `EPSG:4326`).
- **Trazabilidad y agrupación:** `Polygon_ID` (identificador único del polígono de origen).
- **Diez bandas espectrales predictoras:** `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`.
- **Etiquetas de cobertura:** `Class_ID` (entero `1` a `5`) y `Class` (etiqueta categórica). Para el entrenamiento de los clasificadores se utilizan exclusivamente las diez bandas espectrales como matriz $X$, evitando que el modelo memorice posiciones geográficas.

### 3.3 Registros reales y distribución por clase del archivo exportado

| Polygon_ID | Clase real (`Class`) | B02 | B04 | B08 | B11 | B12 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `poly_sue_001` | Suelo Desnudo y Roca | 0,03066 | 0,05546 | 0,30171 | 0,21701 | 0,10591 |
| `poly_bos_001` | Bosque | 0,02410 | 0,02450 | 0,43349 | 0,21077 | 0,08741 |
| `poly_pas_001` | Pastos y Cultivos | 0,03888 | 0,05409 | 0,27997 | 0,21045 | 0,10816 |
| `poly_urb_002` | Urbano y Vías | 0,08691 | 0,09114 | 0,16153 | 0,22448 | 0,20800 |
| `poly_agu_001` | Agua | 0,05982 | 0,08268 | 0,10294 | 0,11185 | 0,07525 |

*Tabla 1. Muestra de filas reales de `dataset.tsv` (cinco bandas de diez mostradas por espacio). El archivo íntegro conserva las 15 columnas y huella SHA-256 `f90bc65884ea46f5f7c989ff00cd4d7f8d8026aa948d504a7b0b03a42def001a` [3].*

| Class_ID | Clase de cobertura | Polígonos efectivos | Píxeles en Desarrollo | Píxeles en Prueba | Total píxeles (`dataset.tsv`) | Proporción (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Bosque** | 12 | 6.194 | 1.222 | **7.416** | 52,39 % |
| **3** | **Suelo Desnudo y Roca** | 115 | 3.679 | 1.189 | **4.868** | 34,39 % |
| **4** | **Urbano y Vías** | 5 | 800 | 71 | **871** | 6,15 % |
| **2** | **Pastos y Cultivos** | 8 | 636 | 203 | **839** | 5,93 % |
| **5** | **Agua** | 5 | 138 | 23 | **161** | 1,14 % |
| — | **Total general** | **145** | **11.447** | **2.708** | **14.155** | **100,00 %** |

*Tabla 2. Distribución de observaciones y polígonos efectivos por cobertura terrestre entre los conjuntos de desarrollo y prueba independiente.*

*(Ver Figura 3: Distribución espacial de las 14.155 muestras etiquetadas del TSV sobre el RGB original).*

---

## 4. Validación espacial e implementación de las cinco arquitecturas (Fase 2)

### 4.1 Separación espacial sin fuga de polígonos (*Zero Spatial Leakage*)
Por autocorrelación espacial (Primera Ley de Tobler), los píxeles contiguos dentro de un mismo polígono comparten iluminación, pendiente y firma espectral muy correlacionada. Si se realizara una partición aleatoria por fila (`train_test_split` simple), píxeles vecinos del mismo polígono caerían simultáneamente en entrenamiento y prueba, inflando artificialmente las métricas por memorización local. Por ello, la unidad atómica de partición definida en [`src/spatial_split.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/spatial_split.py) es `Polygon_ID`.

| Conjunto | Polígonos | Muestras (píxeles) | Proporción | Función dentro del protocolo experimental |
| :--- | :---: | :---: | :---: | :--- |
| **Desarrollo (Dev)** | 116 | 11.447 | 80,87 % | Entrenamiento y optimización de hiperparámetros (`GridSearchCV` en 4 *folds* espaciales). |
| **Prueba independiente (Test)** | 29 | 2.708 | 19,13 % | Evaluación final una sola vez tras fijar los hiperparámetros óptimos. |
| **Total** | **145** | **14.155** | **100,00 %** | Cinco clases con reflectancias válidas en diez bandas. |

La partición quedó congelada en [`data/training/spatial_partitions.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/spatial_partitions.json) con semilla `42`, listando explícitamente `test_polygons`, `dev_polygons` y los cuatro conjuntos `train_polygons`/`val_polygons` [4]. Ningún polígono de prueba participa en el ajuste ni en la búsqueda de hiperparámetros.

*(Ver Figura 4: Distribución espacial de los 145 polígonos efectivos separados en 116 de desarrollo y 29 de prueba).*

### 4.2 Optimización de Hiperparámetros (HPO) y descripción de las cinco arquitecturas
En cumplimiento de las Secciones 3.2 y 3.3 del enunciado, se implementaron y optimizaron cinco arquitecturas de familias matemáticas distintas (`src/model_logistic_regression.py`, `src/model_classical.py` y `src/model_ann.py`). Todos los modelos emplean los cuatro *folds* espaciales oficiales y `GridSearchCV` con `scoring="f1_macro"`, evitando favorecer clasificadores que ignoren las coberturas minoritarias:

1. **Regresión Logística (Baseline obligatorio):** Clasificador lineal discriminativo integrado en un `Pipeline` con `StandardScaler` y penalización $L_2$. Se exploró $C \in \{0,01; 0,1; 1,0; 10,0\}$ y `class_weight` $\in \{\text{None}, \text{'balanced'}\}$. **Configuración óptima:** `C = 1.0`, `class_weight = 'balanced'`, `max_iter = 1000`.
2. **Árbol de Decisión (CART):** Modelo jerárquico no paramétrico basado en particiones ortogonales por reducción de impureza de Gini. Se exploró `max_depth` $\in \{5, 8, 12, \text{None}\}$, `min_samples_leaf` $\in \{1, 2, 5\}$ y `class_weight` $\in \{\text{None}, \text{'balanced'}\}$. **Configuración óptima:** `max_depth = 8`, `min_samples_leaf = 2`, `class_weight = 'balanced'`.
3. **Support Vector Machines (SVM RBF):** Clasificador de margen máximo con proyección no lineal mediante núcleo gaussiano (*Radial Basis Function*) precedido por `StandardScaler`. Se exploró $C \in \{0,1; 1,0; 10,0\}$, $\gamma \in \{\text{'scale'}, \text{'auto'}\}$ y `class_weight`. **Configuración óptima:** `kernel = 'rbf'`, `C = 10.0`, `gamma = 'scale'`.
4. **K-Nearest Neighbors (KNN):** Método no paramétrico basado en instancia y voto ponderado en el espacio euclidiano de 10 dimensiones estandarizadas (`StandardScaler`). Se exploró `n_neighbors` $\in \{3, 5, 11, 15\}$ y `weights` $\in \{\text{'uniform'}, \text{'distance'}\}$. **Configuración óptima:** `n_neighbors = 15`, `weights = 'distance'`.
5. **Red Neuronal Artificial (ANN / MLPClassifier):** Perceptrón multicapa totalmente conectado con activación ReLU, optimizador Adam y regularización $L_2$ precedido por `StandardScaler`. Se evaluaron seis configuraciones combinando arquitecturas ocultas $\{(64,), (128, 64), (64, 32)\}$ y $\alpha \in \{0,0001; 0,001\}$. **Configuración óptima:** `hidden_layer_sizes = (64, 32)`, `alpha = 0.0001`, `max_iter = 500`.

### 4.3 Importancia de bandas espectrales en el Árbol de Decisión
La Sección 3.2 del enunciado destaca el valor de los árboles de decisión para identificar cuáles bandas espectrales son más influyentes al discriminar las coberturas terrestres. A partir del estimador óptimo serializado (`models/decision_tree.joblib`), se extrajo la importancia relativa por reducción normalizada del índice de Gini en los nodos de partición:

| Rango | Banda | Región espectral | Importancia relativa (Gini) | Porcentaje (%) | Interpretación biofísica en el árbol de decisión |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 1 | **`B08`** | NIR amplio (842 nm) | 0,3390 | **33,90 %** | División primaria entre dosel fotosintéticamente activo (alta reflectancia celular) y superficies inertes/agua. |
| 2 | **`B04`** | Rojo visible (665 nm) | 0,3258 | **32,58 %** | Absorción por clorofila; combinada con `B08` reproduce el gradiente físico del NDVI para separar bosque de suelo expuesto. |
| 3 | **`B02`** | Azul visible (490 nm) | 0,2418 | **24,18 %** | Discriminación entre cuerpos de agua, cubiertas urbanas de alta reflectancia visible y sustrato mineral. |
| 4 | **`B11`** | SWIR 1 (1610 nm) | 0,0442 | **4,42 %** | Contraste de humedad entre pasturas/suelo húmedo y roca expuesta o infraestructura vial. |
| 5 | **`B05`** | Red Edge 1 (705 nm) | 0,0231 | **2,31 %** | Refinamiento de bordes entre bosque secundario y pastos/cultivos. |
| 6 | **`B8A`** | NIR estrecho (865 nm) | 0,0154 | **1,54 %** | Estabilización de umbrales infrarrojos en laderas con variación de iluminación. |
| 7 | **`B12`** | SWIR 2 (2190 nm) | 0,0086 | **0,86 %** | Separación fina en suelos minerales y materiales construidos. |
| 8–10 | **`B03`, `B07`, `B06`** | Verde / Red Edge 2–3 | 0,0020 | **0,20 %** | Información colineal ya capturada por las divisiones superiores en `B08`, `B04` y `B05`. |

*Tabla 3. Importancia de variables espectrales del Árbol de Decisión (`outputs/figures/decision_tree_band_importance.png`).*

Las bandas **`B08` (33,90 %)** y **`B04` (32,58 %)** concentran el **66,48 %** de la reducción de impureza porque constituyen los dos polos espectrales del vigor vegetal, mientras que **`B02` (24,18 %)** junto con **`B11` (SWIR 1, 4,42 %)** y **`B05` (Red Edge 1, 2,31 %)** resuelven las bifurcaciones secundarias entre agua, estructuras antrópicas, pastos y suelo desnudo.

![Importancia de bandas - Árbol de Decisión](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/decision_tree_band_importance.png)  
*Figura 5. Importancia relativa de las diez bandas Sentinel-2 en el Árbol de Decisión óptimo (`max_depth=8`). Fuente: `models/decision_tree.joblib` y `outputs/figures/decision_tree_band_importance.png`.*

---

## 5. Comparación de arquitecturas y evaluación de desempeño

Se evaluaron las cinco arquitecturas sobre la misma partición de prueba independiente ($N = 2.708$ píxeles en 29 polígonos). Los resultados reportados en la Tabla 4 corresponden exactamente a los artefactos auditables en `outputs/metrics/*.json` y `outputs/metrics/comparison_table.md` [6].

| Modelo | Overall Accuracy (test) | Aciertos / Total | Kappa de Cohen | Macro-F1 (test) | Weighted-F1 (test) | CV Macro-F1 (4 folds dev) | Tiempo HPO (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN / MLP** `(64, 32)` | 88,74 % | 2.403 / 2.708 | 0,8146 | **0,7075** | 0,8929 | **0,7103 ± 0,0866** | 103,64 s |
| **SVM (RBF)** `C=10` | **89,70 %** | **2.429 / 2.708** | **0,8283** | 0,6889 | **0,8965** | 0,6538 ± 0,0830 | 7,32 s |
| **Árbol de Decisión** `depth=8` | 86,41 % | 2.340 / 2.708 | 0,7827 | 0,6794 | 0,8795 | 0,6926 ± 0,0923 | 7,51 s |
| **Regresión Logística** (Baseline) | 82,87 % | 2.244 / 2.708 | 0,7343 | 0,6516 | 0,8505 | 0,7039 ± 0,0749 | 5,81 s |
| **KNN** `k=15, distance` | 87,70 % | 2.375 / 2.708 | 0,7936 | 0,6159 | 0,8708 | 0,6805 ± 0,0961 | 4,17 s |

*Tabla 4. Tabla resumen comparativa de las cinco arquitecturas de Machine Learning ordenadas por `Macro-F1` en prueba independiente [6].*

### 5.1 Significado e interpretación de las métricas globales
- **Overall Accuracy (Exactitud Global):** Proporción total de píxeles clasificados correctamente ($\text{OA} = \frac{1}{N}\sum_{i} M_{ii}$, con $N = 2.708$). Al estar dominado el conjunto de prueba por Bosque (1.222 filas) y Suelo Desnudo (1.189 filas), que suman el 89,03 % del test, un modelo puede alcanzar alto Accuracy fallando en coberturas minoritarias.
- **Coeficiente Kappa de Cohen ($\kappa$):** Mide el acuerdo inter-clasificador descontando la concordancia atribuible al azar según las distribuciones marginales. Tanto SVM ($\kappa = 0,8283$) como ANN ($\kappa = 0,8146$) se ubican en el rango de concordancia excelente ($> 0,80$).
- **Precision, Recall y Macro-F1:** Para cada categoría $c$, la **Precisión** ($\text{TP}_c / (\text{TP}_c + \text{FP}_c)$) cuantifica la confiabilidad de la predicción (ausencia de comisión/falsos positivos) y el **Recall o Exhaustividad** ($\text{TP}_c / (\text{TP}_c + \text{FN}_c)$) mide la completitud de detección (ausencia de omisión/falsos negativos). El **F1-Score** es su media armónica, y el **Macro-F1** promedia sin ponderación las cinco clases ($1/5 \sum F1_c$), penalizando severamente cualquier colapso en Agua, Urbano o Pastos.

### 5.2 Desempeño detallado por clase: Precision, Recall y F1-Score en los cinco modelos
En cumplimiento estricto de la Sección 3.3 del enunciado (*Precision, Recall and F1-Score for each land-cover category*), la Tabla 5 sintetiza las métricas por clase sobre las 2.708 muestras de prueba para las cinco arquitecturas:

| Clase de cobertura (`N test`) | Métrica | Reg. Logística | Árbol Decisión | SVM (RBF) | KNN (`k=15`) | ANN / MLP | Análisis comparativo por cobertura |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Agua**<br>(`n = 23`) | **Precision**<br>**Recall**<br>**F1-Score** | 0,4510<br>**1,0000**<br>0,6216 | 0,4510<br>**1,0000**<br>0,6216 | **0,4792**<br>**1,0000**<br>**0,6479** | 0,4694<br>**1,0000**<br>0,6389 | **0,4792**<br>**1,0000**<br>**0,6479** | Los 5 modelos detectan el 100 % de los píxeles de Agua (`23/23`), pero cometen entre 25 y 28 falsos positivos desde Urbano/Vías oscuras. SVM y ANN empatan en el máximo F1 (0,6479). |
| **Bosque**<br>(`n = 1.222`) | **Precision**<br>**Recall**<br>**F1-Score** | 0,9984<br>**1,0000**<br>0,9992 | **1,0000**<br>**1,0000**<br>**1,0000** | 0,9992<br>**1,0000**<br>0,9996 | 0,9992<br>**1,0000**<br>0,9996 | 0,9992<br>**1,0000**<br>0,9996 | Firma espectral altamente separable en NIR/Rojo. Los 5 modelos recuperan los 1.222 píxeles sin omisiones (F1 $\ge 0,9992$). |
| **Pastos y Cultivos**<br>(`n = 203`) | **Precision**<br>**Recall**<br>**F1-Score** | 0,3251<br>**0,7094**<br>**0,4458** | 0,3239<br>0,5665<br>0,4122 | **0,4868**<br>0,3645<br>0,4169 | 0,3793<br>0,1626<br>0,2276 | 0,3658<br>0,4631<br>0,4087 | Clase de transición espectral con suelo ralo. LR logra alto recall (70,94 %) a costa de baja precisión (299 falsos positivos desde Suelo). SVM logra la mejor precisión (48,68 %). |
| **Suelo Desnudo y Roca**<br>(`n = 1.189`) | **Precision**<br>**Recall**<br>**F1-Score** | **0,9599**<br>0,6838<br>0,7986 | 0,9514<br>0,7897<br>0,8631 | 0,9073<br>**0,8974**<br>**0,9023** | 0,8763<br>0,8940<br>0,8851 | 0,9176<br>0,8612<br>0,8885 | SVM alcanza el mejor equilibrio (F1: 0,9023; 1.067 aciertos) seguido por ANN (F1: 0,8885; 1.024 aciertos), superando ampliamente el recall del baseline lineal (68,38 %). |
| **Urbano y Vías**<br>(`n = 71`) | **Precision**<br>**Recall**<br>**F1-Score** | 0,2937<br>0,5915<br>0,3925 | 0,4409<br>0,5775<br>0,5000 | 0,3945<br>**0,6056**<br>0,4778 | 0,2500<br>0,4789<br>0,3285 | **0,6250**<br>0,5634<br>**0,5926** | **Ventaja decisiva de ANN/MLP:** alcanza una precisión de **62,50 %** y F1 de **0,5926** al cometer solo 24 falsos positivos urbanos, frente a 66 de SVM y 101 de Regresión Logística. |

*Tabla 5. Métricas detalladas de Precisión, Exhaustividad (Recall) y F1-Score por clase en el conjunto de prueba independiente ($N = 2.708$) para las cinco arquitecturas, extraídas directamente de `outputs/metrics/*.json` [6–9].*

---

## 6. Análisis de matrices de confusión de las cinco arquitecturas

Las filas de todas las matrices representan las **clases reales** y las columnas las **clases predichas**, en el orden alfabético estandarizado: `Agua`, `Bosque`, `Pastos y Cultivos`, `Suelo Desnudo y Roca`, `Urbano y Vías` ($N = 2.708$).

### 6.1 Regresión Logística (Baseline lineal): `OA = 82,87 %` · `Macro-F1 = 0,6516`
El clasificador lineal reconoce los 1.222 píxeles de Bosque y los 23 de Agua, pero su frontera hiperplanar con `class_weight='balanced'` sobre-extiende las clases minoritarias: etiqueta como Pastos y Cultivos **299 píxeles reales de Suelo Desnudo y Roca** y como Urbano y Vías otros **75 píxeles de Suelo**, reduciendo el recall de Suelo Desnudo a 68,38 % (813/1.189). Asimismo, asigna a Agua 28 píxeles de Urbano y Vías, dejando la precisión de Agua en 45,10 %.

### 6.2 Árbol de Decisión (CART, `max_depth=8`): `OA = 86,41 %` · `Macro-F1 = 0,6794`
El árbol jerárquico eleva los aciertos totales a **2.340 de 2.708** ($\kappa = 0,7827$). Logra separación perfecta en Bosque (`F1 = 1,0000`, cero falsos positivos y cero falsos negativos) y recupera **939 de 1.189 píxeles de Suelo Desnudo** (precisión de 95,14 %). Además, reduce drásticamente los falsos positivos en Urbano y Vías respecto al baseline (solo 10 píxeles de Suelo confundidos con Urbano frente a 75 en Regresión Logística), llevando el F1 urbano a **0,5000** (tercer mejor registro del estudio). Su limitación principal sigue siendo la confusión de 240 píxeles de Suelo Desnudo hacia Pastos y Cultivos por particiones rectangulares sobre `B08` y `B04`.

![Matriz de confusión - Árbol de Decisión](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/decision_tree_confusion_matrix.png)  
*Figura 6. Matriz de confusión del Árbol de Decisión en el test reservado: 2.340 aciertos de 2.708 (86,41 %). Fuente: `outputs/metrics/decision_tree.json`.*

### 6.3 Support Vector Machines (SVM RBF): `OA = 89,70 %` · `Macro-F1 = 0,6889`
SVM con núcleo RBF obtiene la mayor exactitud global del estudio (**2.429 aciertos de 2.708**) y el mayor Kappa (**0,8283**). Eleva la detección de Suelo Desnudo y Roca a **1.067 aciertos** (recall: 89,74 %, precisión: 90,73 %, F1: **0,9023**) y acierta 43 de 71 píxeles de Urbano y Vías. No obstante, solo recupera 74 de los 203 píxeles reales de Pastos y Cultivos (confundiendo 106 con Suelo Desnudo) y comete 66 falsos positivos hacia Urbano y Vías (23 desde Pastos y 43 desde Suelo), lo que explica que su Macro-F1 (0,6889) sea superado por la red neuronal.

### 6.4 K-Nearest Neighbors (KNN, `k=15, distance`): `OA = 87,70 %` · `Macro-F1 = 0,6159`
KNN ilustra con claridad la diferencia entre Exactitud Global y Macro-F1: obtiene el **tercer mejor Overall Accuracy (87,70 %, 2.375 aciertos)** gracias a su sólido desempeño en las dos clases masivas —Bosque (1.222/1.222) y Suelo Desnudo (1.063/1.189, F1: 0,8851)—, pero registra el **Macro-F1 más bajo (0,6159)**. En el espacio euclidiano de 10 bandas, la vecindad de $k = 15$ diluye las clases de menor soporte o geometría fragmentada: solo recupera **33 de 203 píxeles de Pastos y Cultivos** (recall: 16,26 %, desplazando 139 hacia Suelo Desnudo) y comete 102 falsos positivos hacia Urbano y Vías (precisión urbana de apenas 25,00 %).

![Matriz de confusión - KNN](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/knn_confusion_matrix.png)  
*Figura 7. Matriz de confusión de KNN en el test reservado: 2.375 aciertos de 2.708 (87,70 %). Fuente: `outputs/metrics/knn.json`.*

### 6.5 Red Neuronal Artificial (ANN / MLP `(64, 32)`): `OA = 88,74 %` · `Macro-F1 = 0,7075`
La arquitectura neuronal multicapa se propuso como modelo campeón para la inferencia por haber obtenido el **mayor Macro-F1 promedio en validación cruzada sobre desarrollo (0,7103 ± 0,0866)**, criterio metodológico definido antes de abrir el conjunto de prueba. Al evaluarse sobre el test independiente confirmó el **mayor Macro-F1 del proyecto (0,7075)** con **2.403 aciertos (88,74 %)** y $\kappa = 0,8146$. Su ventaja sobre SVM radica en la **selectividad sobre Urbano y Vías**: solo clasifica erróneamente **1 píxel de Suelo Desnudo como Urbano** (frente a 43 en SVM y 71 en KNN), elevando la precisión urbana a **62,50 %** y el F1 de esa clase a **0,5926** (+11,48 puntos porcentuales sobre SVM), además de recuperar 94 píxeles de Pastos y Cultivos frente a 74 de SVM.

---

## 7. Inferencia ráster sobre la escena completa y cuantificación de áreas (Fase 3)

En cumplimiento de la Sección 3.4 del enunciado (*Final Raster Prediction, Spatial Consistency, Thematic Visualization and Area Quantification*), los cinco modelos entrenados fueron desplegados sobre el stack multiespectral completo mediante los scripts [`src/predict_raster.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/predict_raster.py) y [`src/predict_raster_hania.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/predict_raster_hania.py).

### 7.1 Consistencia espacial y dominio común verificado de diez bandas
La grilla canónica de `stack_10bands.tif` posee dimensiones de **`1.054 × 681` píxeles** (717.774 celdas totales) en el CRS **`EPSG:9377`** con transformación afín de **$10,0\text{ m} \times 10,0\text{ m}$** ($100\text{ m}^2 = 0,01\text{ ha} = 0,0001\text{ km}^2$ por píxel). Si bien la máscara general `mask_valid.tif` habilita 471.579 píxeles libres de nubes en las bandas visibles de 10 m, las bandas *Red Edge* y NIR estrecho (`B05`, `B06`, `B07`, `B8A`) presentan bordes con `NoData` en los márgenes del recorte. Para garantizar rigor físico y evitar la predicción sobre vectores incompletos o reflectancias imputadas sintéticamente, la inferencia se ejecutó sobre la intersección estricta donde las diez bandas y la máscara oficial contienen datos finitos válidos: **exactamente 342.525 píxeles clasificables (3.425,25 ha = 34,2525 $\text{km}^2$)**, asignando `NoData = 0` a las 375.249 celdas restantes.

Todos los GeoTIFF categóricos de salida (`outputs/rasters/landcover_*.tif`, tipo `uint8`, compresión `deflate`, banda única con códigos `1` a `5`) conservan intactos el CRS `EPSG:9377` y la matriz de transformación afín de la imagen Sentinel-2 original.

### 7.2 Cuantificación oficial de áreas por cobertura terrestre (ANN/MLP verificado)
Se auditó directamente el archivo binario [`outputs/rasters/landcover_ann.tif`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/landcover_ann.tif) (huella SHA-256: `540cb9e738acff655df9d2416c1896f0f4df7403e8f4b41002a7802fe44ba0c0`) junto con [`outputs/metrics/inferencia_10bandas/area_stats_ann.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/inferencia_10bandas/area_stats_ann.json), confirmando una **coincidencia exacta del 100 %** entre la matriz GeoTIFF y las superficies reportadas en la leyenda cartográfica de QGIS:

| Class_ID | Cobertura terrestre | Color simbología (`.qml`) | Píxeles clasificados (`landcover_ann.tif`) | Área en hectáreas ($\text{ha}$) | Área en kilómetros cuadrados ($\text{km}^2$) | Porcentaje del dominio (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Bosque** | `#1b7837` (Verde oscuro) | 218.951 | **2.189,51 ha** | **21,8951 $\text{km}^2$** | **63,92 %** |
| **3** | **Suelo Desnudo y Roca** | `#d73027` (Rojo) | 117.213 | **1.172,13 ha** | **11,7213 $\text{km}^2$** | **34,22 %** |
| **2** | **Pastos y Cultivos** | `#a6d96a` (Verde claro) | 3.822 | **38,22 ha** | **0,3822 $\text{km}^2$** | **1,12 %** |
| **4** | **Urbano y Vías** | `#e08214` (Naranja) | 1.993 | **19,93 ha** | **0,1993 $\text{km}^2$** | **0,58 %** |
| **5** | **Agua** | `#2166ac` (Azul) | 546 | **5,46 ha** | **0,0546 $\text{km}^2$** | **0,16 %** |
| — | **Total clasificado (10 bandas)** | — | **342.525** | **3.425,25 ha** | **34,2525 $\text{km}^2$** | **100,00 %** |

*Tabla 6. Cuantificación espacial definitiva de coberturas terrestres en la escena de San José del Palmar mediante el modelo seleccionado ANN/MLP (`outputs/rasters/landcover_ann.tif` y `outputs/metrics/inferencia_10bandas/area_stats_ann.json`).*

### 7.3 Sensibilidad espacial: comparación de áreas estimadas entre las cinco arquitecturas
Para evaluar la estabilidad territorial de las predicciones en la escena completa (`342.525` píxeles), la Tabla 7 compara las superficies en hectáreas ($\text{ha}$) y porcentaje generadas por los cinco modelos en `outputs/metrics/inferencia_10bandas/area_stats_*.json`:

| Cobertura terrestre | ANN / MLP (ha · %) | SVM RBF (ha · %) | KNN `k=15` (ha · %) | Árbol Decisión (ha · %) | Reg. Logística (ha · %) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Bosque** | **2.189,51 ha** (63,92 %) | **2.180,01 ha** (63,65 %) | 2.223,36 ha (64,91 %) | 1.544,52 ha (45,09 %) | 2.521,82 ha (73,62 %) |
| **2. Pastos y Cultivos** | **38,22 ha** (1,12 %) | **37,26 ha** (1,09 %) | 23,92 ha (0,70 %) | 80,48 ha (2,35 %) | 254,97 ha (7,44 %) |
| **3. Suelo Desnudo y Roca** | **1.172,13 ha** (34,22 %) | **1.180,54 ha** (34,47 %) | 1.148,70 ha (33,54 %) | 1.768,50 ha (51,63 %) | 616,46 ha (18,00 %) |
| **4. Urbano y Vías** | **19,93 ha** (0,58 %) | **22,23 ha** (0,65 %) | 22,95 ha (0,67 %) | 25,70 ha (0,75 %) | 25,98 ha (0,76 %) |
| **5. Agua** | **5,46 ha** (0,16 %) | **5,21 ha** (0,15 %) | 6,32 ha (0,18 %) | 6,05 ha (0,18 %) | 6,02 ha (0,18 %) |
| **Total escena válida** | **3.425,25 ha** (100 %) | **3.425,25 ha** (100 %) | **3.425,25 ha** (100 %) | **3.425,25 ha** (100 %) | **3.425,25 ha** (100 %) |

*Tabla 7. Comparación de superficies clasificadas (en hectáreas y porcentaje) por las cinco arquitecturas sobre el mismo dominio válido de 342.525 píxeles (`outputs/metrics/inferencia_10bandas/`).*

**Interpretación espacial y coherencia entre modelos:**
1. **Alta convergencia entre clasificadores no lineales suaves (ANN, SVM y KNN):** Las tres arquitecturas coinciden estrechamente en la estructura territorial del cañón del río Ingará, estimando entre **63,65 % y 64,91 % de Bosque** (`2.180`–`2.223 ha`), entre **33,54 % y 34,47 % de Suelo Desnudo y Roca** (`1.148`–`1.180 ha`), y entre **5,21 y 6,32 ha de cuerpos de agua**. La diferencia entre ANN y SVM en Bosque y Suelo Desnudo es inferior al **0,27 %** del área total ($< 9,5\text{ ha}$), lo que confirma una alta robustez cartográfica.
2. **Sesgo de modelos lineales y ortogonales en inferencia completa:** La Regresión Logística subestima severamente el Suelo Desnudo (`616,46 ha`, 18,00 %) y multiplica por seis el área de Pastos y Cultivos (`254,97 ha`), reproduciendo a escala de escena el mismo error observado en su matriz de confusión (donde desvió 299 píxeles de Suelo hacia Pastos). Por su parte, el Árbol de Decisión sobre-asigna Suelo Desnudo (`1.768,50 ha`, 51,63 %) en laderas sombreadas debido a la rigidez de sus cortes axiales sobre `B08` y `B04`.

### 7.4 Visualización temática en QGIS
El archivo `outputs/rasters/landcover_ann.tif` se integró en el proyecto editable de QGIS 3.44.13 ([`outputs/rasters/Clasificacion_Cobertura_ANN.qgz`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/Clasificacion_Cobertura_ANN.qgz)) aplicando la simbología discreta oficial ([`outputs/rasters/landcover_style.qml`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/landcover_style.qml)). El mapa final exportado en [`outputs/figures/Mapa de cobertura terrestre - ANN.png`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/Mapa%20de%20cobertura%20terrestre%20-%20ANN.png) y [`outputs/figures/Mapa de cobertura ANN.pdf`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/Mapa%20de%20cobertura%20ANN.pdf) (Figura 8 / Anexo A) permite identificar la matriz dominante de bosque húmedo tropical (63,92 %), los corredores aluviales y laderas con suelo desnudo o cicatrices de remoción en masa (34,22 %), las áreas de pasturas (1,12 %), el trazado vial y núcleo urbano de San José del Palmar (0,58 %) y el cauce visible del río Ingará (0,16 %).

![Mapa de cobertura terrestre - ANN/MLP](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/Mapa%20de%20cobertura%20terrestre%20-%20ANN.png)  
*Figura 8. Mapa temático final de clasificación de cobertura terrestre generado con la Red Neuronal Artificial (ANN/MLP) sobre los 342.525 píxeles válidos en `EPSG:9377` y maquetado en QGIS 3.44.13. Las cifras de superficie de la leyenda coinciden al 100 % con el conteo de píxeles de `outputs/rasters/landcover_ann.tif`.*

---

## 8. Discusión general, limitaciones y conclusiones

### 8.1 Hallazgos principales
- **Cumplimiento integral del pipeline GeoAI:** Se completó la transformación desde reflectancias satelitales Sentinel-2 L2A (10 bandas a 10 m en `EPSG:9377`) hacia un dataset tabular verificable (`14.155` filas), el entrenamiento y optimización de cinco arquitecturas de aprendizaje automático y el despliegue espacial de vuelta a GeoTIFF categórico con cuantificación de áreas.
- **Control de fuga espacial:** El protocolo de separación por `Polygon_ID` (116 polígonos de desarrollo y 29 de prueba) y cuatro *folds* compartidos evita la inflación artificial de métricas por vecindad inmediata dentro de un mismo polígono.
- **Complementariedad entre métricas y modelos:** No existe un clasificador superior en todas las dimensiones simultáneamente. **SVM RBF** maximiza la exactitud global (**89,70 %**) y Kappa (**0,8283**) por su alto acierto en Suelo Desnudo (F1: 0,9023), mientras que **ANN/MLP** maximiza el **Macro-F1 tanto en validación cruzada (0,7103) como en prueba independiente (0,7075)** al reducir drásticamente los falsos positivos en la clase minoritaria Urbano y Vías (F1: 0,5926 vs. 0,4778 en SVM).
- **Interpretabilidad física:** El Árbol de Decisión demostró que el par `B08` (NIR) y `B04` (Rojo) aporta el 66,48 % del poder discriminante primario, complementado por `B02` (Azul, 24,18 %) y `B11`/`B05` (SWIR/Red Edge, 6,73 %) para desambiguar agua, infraestructura y suelo mineral.

### 8.2 Limitaciones metodológicas y espaciales
1. **Soporte reducido de clases minoritarias en prueba:** Debido a que la partición respeta unidades poligonales completas y el paisaje chocoano está fuertemente dominado por bosque y laderas intervenidas, el conjunto de prueba reservado cuenta con un polígono de Agua (23 píxeles) y uno de Urbano y Vías (71 píxeles). Aunque `Macro-F1` visibiliza el comportamiento en estas clases, la generalización hacia otros cascos urbanos o cauces requiere ampliar el inventario vectorial de polígonos independientes.
2. **Autocorrelación residual entre polígonos próximos:** Separar por `Polygon_ID` elimina la fuga intra-polígono, pero polígonos cercanos sobre una misma vertiente aún pueden compartir condiciones atmosféricas y topográficas similares.
3. **Alcance de la clasificación mono-temporal post-sismo:** Las **1.172,13 ha** clasificadas como Suelo Desnudo y Roca por la ANN (y **1.180,54 ha** por SVM) reflejan la superficie total desprovista de vegetación observada el 18 de agosto de 2026 (que incluye cicatrices cosísmicas, afloramientos rocosos, playones aluviales y suelos previamente descubiertos), por lo que no deben reportarse en su totalidad como daño neto exclusivo del sismo sin restar una línea base pre-evento.

### 8.3 Conclusión
El flujo de trabajo desarrollado demuestra que las arquitecturas no lineales con estandarización de características (**ANN/MLP** y **SVM RBF**) superan de manera consistente tanto al baseline lineal (**Regresión Logística**, +5,87 a +6,83 puntos en Accuracy y +0,0373 a +0,0559 en Macro-F1) como a los métodos locales o axiales (**KNN** y **Árbol de Decisión**) al clasificar firmas multiespectrales de 10 bandas en terrenos montañosos tropicales. La selección de **ANN/MLP** mediante validación cruzada espacial (`CV Macro-F1 = 0,7103`) se validó de forma independiente en prueba (`Test Macro-F1 = 0,7075`) y permitió cartografiar de forma reproducible las **3.425,25 ha** del dominio común de diez bandas en San José del Palmar.

---

## 9. Fuentes y artefactos de verificación

- **[1]** Servicio Geológico Colombiano (10 ago. 2026). *Actualización sobre el sismo de San José del Palmar, Chocó*. https://www2.sgc.gov.co/Noticias/Paginas/SGC-actualiza-la-informacion-sobre-el-sismo-ocurrido-en-San-Jose-del-Palmar-Choco.aspx
- **[2]** Repositorio `EmergentesFinal`: [`config/project.yaml`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/config/project.yaml); [`src/01_prepare_stack.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/01_prepare_stack.py); [`src/02_extract_tsv.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/02_extract_tsv.py). https://github.com/Nicolas11Leon/EmergentesFinal
- **[3]** Dataset tabular [`data/training/dataset.tsv`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/dataset.tsv) (14.155 muestras, 15 columnas, SHA-256: `f90bc65884ea46f5f7c989ff00cd4d7f8d8026aa948d504a7b0b03a42def001a`).
- **[4]** Particiones espaciales [`data/training/spatial_partitions.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/spatial_partitions.json) (116 polígonos dev, 29 test, 4 folds CV, semilla 42).
- **[5]** Scripts de entrenamiento y HPO: [`src/model_logistic_regression.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/model_logistic_regression.py), [`src/model_classical.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/model_classical.py) y [`src/model_ann.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/model_ann.py).
- **[6]** Consolidado de métricas: [`outputs/metrics/comparison_table.md`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/comparison_table.md) y archivos JSON por modelo en [`outputs/metrics/`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/).
- **[7]** Métricas y matrices por arquitectura: [`logistic_regression.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/logistic_regression.json), [`decision_tree.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/decision_tree.json), [`svm.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/svm.json), [`knn.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/knn.json) y [`ann.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/ann.json).
- **[8]** Inferencia ráster espacial y estadísticas de superficie verificadas (342.525 píxeles): [`src/predict_raster_hania.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/predict_raster_hania.py), [`outputs/rasters/landcover_ann.tif`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/landcover_ann.tif) (SHA-256: `540cb9e738acff655df9d2416c1896f0f4df7403e8f4b41002a7802fe44ba0c0`) y [`outputs/metrics/inferencia_10bandas/`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/metrics/inferencia_10bandas/).
- **[9]** Proyecto cartográfico QGIS 3.44.13: [`outputs/rasters/Clasificacion_Cobertura_ANN.qgz`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/Clasificacion_Cobertura_ANN.qgz), simbología [`outputs/rasters/landcover_style.qml`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/landcover_style.qml) y exportación final [`outputs/figures/Mapa de cobertura terrestre - ANN.png`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/Mapa%20de%20cobertura%20terrestre%20-%20ANN.png).
