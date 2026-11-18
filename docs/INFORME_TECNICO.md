# Proyecto Final: Análisis Comparativo de Arquitecturas de Machine Learning para la Clasificación de Cobertura de la Tierra basada en Datos Satelitales Sentinel-2

**Curso:** Tecnologías Emergentes  
**Profesor:** Andrés Oswaldo Calderón Romero, Ph.D.  
**Institución:** Pontificia Universidad Javeriana  
**Fecha:** Noviembre de 2026  

---

### Autores y Distribución de Roles

| Estudiante | Rol Asignado | Responsabilidad Principal |
| :--- | :--- | :--- |
| **Nicolás León** | **Rol 1** | *Data Engineering & Baseline:* Preparación del stack de 10 bandas (`EPSG:9377`), muestreo de polígonos, extracción tabular a `dataset.tsv`, auditoría estricta de calidad, particiones espaciales y Modelo 1 (Regresión Logística con HPO). |
| **Andrés** | **Rol 2** | *Modelos Clásicos & Validación:* Implementación de Decision Tree, Support Vector Machines (SVM RBF) y K-Nearest Neighbors (KNN), optimización de hiperparámetros con GridSearchCV, trazabilidad SHA-256 e interpretabilidad de bandas. |
| **Hania Valeria** | **Rol 3** | *Redes Neuronales & Producto Geoespacial:* Implementación de Red Neuronal Artificial (ANN / MLPClassifier), inferencia de la escena completa a GeoTIFF categórico, cuantificación de áreas ($ha$ y $km^2$) y mapas temáticos en QGIS. |

---

## Resumen Ejecutivo (Abstract)

Este proyecto implementa un flujo de trabajo integral y reproducible de Inteligencia Artificial Geoespacial (GeoAI) para cartografiar y cuantificar las coberturas de la tierra y las afectaciones cosísmicas detonadas por el sismo de magnitud $M_w$ 7.4 ocurrido el 10 de agosto de 2026 en el municipio de San José del Palmar (Chocó, Colombia). Utilizando imágenes multiespectrales Sentinel-2 MSI Nivel-2A (reflectancia de superficie BOA) del 18 de agosto de 2026, se consolidó un stack de 10 bandas espectrales remuestreado a una resolución común de 10 metros en el sistema de coordenadas oficial de Colombia (MAGNA-SIRGAS Origen Nacional, `EPSG:9377`).

A partir de 240 polígonos de verdad de terreno (que integran 181 cicatrices reales de deslizamientos inventariadas tras el evento junto con muestras estratificadas de Bosque, Pastos y Cultivos, Urbano y Vías, y Cuerpos de Agua), se extrajo un dataset tabular de 14.155 píxeles puros. Para eliminar cualquier riesgo de sobreajuste y fuga de información (*zero data leakage*), se implementó una estrategia de validación espacial agrupada por polígono (`GroupKFold`), reservando el 20% de los polígonos para una partición de prueba independiente (2.708 muestras) y el 80% restante (11.447 muestras) estructurado en 4 folds de desarrollo estratificados.

Se entrenaron y optimizaron cuatro arquitecturas de Machine Learning con `GridSearchCV` guiadas por la métrica Macro-F1: (1) Regresión Logística con regularización $L_2$ (Baseline), (2) Árboles de Decisión (CART), (3) Máquinas de Vectores de Soporte con kernel RBF (SVM), y (4) K-Vecinos más Cercanos (KNN), encontrándose en desarrollo la Red Neuronal Artificial (MLP). Los resultados sobre la partición de prueba espacial independiente posicionan a **SVM (RBF)** como el modelo de mayor capacidad de generalización con una **Exactitud Global (Overall Accuracy) del 89.70%**, un **Índice Kappa de 0.8283** y un **Macro-F1 de 0.6889**, seguido por Decision Tree (86.41% OA, 0.6794 F1), KNN (87.70% OA, 0.6159 F1) y Regresión Logística (82.87% OA, 0.6516 F1). La inferencia de la escena completa sobre 471.371 píxeles válidos permitió cuantificar el territorio clasificado en 4.713,71 ha, evidenciando un dominio del bosque húmedo tropical (46.25%) y una fuerte presencia de suelo desnudo y depósitos cosísmicos (52.38% del área despejada analizada).

---

## 1. Introducción y Planteamiento del Problema

La transición de los Sistemas de Información Geográfica (SIG) tradicionales hacia la Inteligencia Artificial Geoespacial (GeoAI) permite automatizar el monitoreo territorial y la evaluación de emergencias mediante arquitecturas de aprendizaje automático desacopladas de interfaces de escritorio. Los eventos sísmicos en la Cordillera Occidental colombiana representan una amenaza compleja: laderas de alta pendiente, meteorización profunda e intensas precipitaciones tropicales se conjugan para que un sismo de profundidad intermedia desencadene cientos de movimientos en masa que represan cauces y bloquean corredores viales vitales.

El 10 de agosto de 2026 a las 12:34 UTC, un sismo de magnitud $M_w$ 7.4 con epicentro cercano a San José del Palmar (Chocó) provocó fallas masivas en las laderas del cañón del río Ingará, aislando a la población y destruyendo coberturas boscosas y agropecuarias. El objetivo del presente trabajo es evaluar de manera comparativa y rigurosa cinco arquitecturas de Machine Learning para discriminar cinco coberturas críticas: (1) Bosque, (2) Pastos y Cultivos, (3) Suelo Desnudo y Roca / Deslizamientos, (4) Urbano y Vías, y (5) Cuerpos de Agua, cumpliendo con los estándares de rigor científico, reproducibilidad en código abierto y ausencia total de fuga espacial de información.

---

## 2. Área de Estudio y Stack Multiespectral Sentinel-2

### 2.1 Región de Interés (ROI)
El área de estudio corresponde a un polígono de 49.7 $\text{km}^2$ ubicado en la vertiente pacífica de la Cordillera Occidental (coordenadas centrales $4.90^\circ\text{N}$, $76.23^\circ\text{W}$). Cubre la cuenca media y baja del cañón del río Ingará, el casco urbano del municipio de San José del Palmar y los asentamientos rurales aledaños.

### 2.2 Sistema de Referencia Espacial y Grilla Canónica
Para garantizar la consistencia métrica y la interoperabilidad con las bases cartográficas oficiales de Colombia (IGAC), todas las capas vectoriales y matriciales fueron reproyectadas al sistema **MAGNA-SIRGAS Origen Nacional (`EPSG:9377`)**. La grilla de referencia posee dimensiones de **$1.054 \times 681$ píxeles** con un tamaño de celda exacto de **$10\text{ m} \times 10\text{ m}$** (origen $X = 4.633.008,53\text{ m}$, $Y = 2.103.785,99\text{ m}$).

### 2.3 Stack Multiespectral de 10 Bandas
A partir de la escena Sentinel-2 MSI Nivel-2A del 18 de agosto de 2026 (`S2C_MSIL2A_20260818T153621_N0512_R068_T18NUL`), adquirida en condiciones de cielo despejado tras la emergencia, se ensambló el stack multiespectral procesado en [`data/processed/stack_10bands.tif`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/processed/stack_10bands.tif). Las bandas nativas de 20 metros fueron remuestreadas bilinealmente a la grilla de 10 metros, preservando los valores físicos de reflectancia de superficie (BOA):

| Índice | Banda | Nombre Espectral | Longitud de Onda Central ($\mu\text{m}$) | Resolución Nativa | Rol Biofísico Principal |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 1 | `B02` | Azul (Blue) | 0.490 | 10 m | Penetración atmosférica, cuerpos de agua y contraste suelo-urbano |
| 2 | `B03` | Verde (Green) | 0.560 | 10 m | Pico de reflectancia vegetal (clorofila), cálculo de NDWI y MNDWI |
| 3 | `B04` | Rojo (Red) | 0.665 | 10 m | Absorción máxima de clorofila, base para NDVI y detección de suelo |
| 4 | `B05` | Red Edge 1 | 0.705 | 20 m $\rightarrow$ 10 m | Pendiente del borde rojo, diagnóstico de estrés y estructura foliar |
| 5 | `B06` | Red Edge 2 | 0.740 | 20 m $\rightarrow$ 10 m | Transición al infrarrojo cercano, discriminación dosel denso vs. rastrojo |
| 6 | `B07` | Red Edge 3 | 0.783 | 20 m $\rightarrow$ 10 m | Meseta del infrarrojo cercano, detección de densidad de biomasa |
| 7 | `B08` | NIR Amplio | 0.842 | 10 m | Alta dispersión en mesófilo foliar, cálculo canónico de NDVI |
| 8 | `B8A` | NIR Angosto | 0.865 | 20 m $\rightarrow$ 10 m | Eliminación de contaminación por vapor de agua atmosférico |
| 9 | `B11` | SWIR 1 | 1.610 | 20 m $\rightarrow$ 10 m | Contenido de humedad de la vegetación y reflectancia de rocas/suelos |
| 10 | `B12` | SWIR 2 | 2.190 | 20 m $\rightarrow$ 10 m | Discriminación geológica, cicatrices secas de deslizamiento y construcciones |

### 2.4 Máscara de Calidad Radiométrica
Se generó la capa [`data/processed/mask_valid.tif`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/processed/mask_valid.tif) mediante la intersección de la capa de clasificación de escena (SCL) decodificada y el umbral de reflectancia azul ($B02 \le 0.10$), asegurando la exclusión estricta de sombras orográficas profundas, cirros delgados y nubes residuales, conservando **471.371 píxeles válidos** (94.8% del área de estudio).

---

## 3. Ingeniería de Datos, Verdad de Terreno y Validación Espacial

### 3.1 Construcción del Dataset y Control de Calidad
La verdad de terreno se consolidó en [`data/vectors/ground_truth.gpkg`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/vectors/ground_truth.gpkg). Contiene:
- **Cicatrices de Deslizamiento (Clase 3: Suelo Desnudo y Roca):** 181 polígonos originales delimitados a partir de la diferencia bitemporal $\Delta\text{NDVI} = \text{NDVI}_{t_2} - \text{NDVI}_{t_1}$ con umbral $\mu - 2\sigma \le -0.0926$, complementados con la firma del infrarrojo cercano $\Delta\text{NIR}$ y fotointerpretación.
- **Coberturas Estables (Clases 1, 2, 4 y 5):** Muestras espaciales independientes de bosque maduro, mosaicos agrícolas y pastos, cabecera municipal de San José del Palmar e infraestructura vial, y cauces hídricos principales del río Ingará, verificadas con índices biofísicos y composiciones en falso color (4-3-2 y SWIR 11-8-4).

El script [`src/02_extract_tsv.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/02_extract_tsv.py) realizó la intersección geométrica entre los polígonos y el stack ráster:
- **Criterio de pureza de firma:** Extracción píxel a píxel con centroide interior al polígono (`all_touched=False`).
- **Filtrado radiométrico:** Se eliminaron 95 polígonos situados sobre los bordes triangulares externos de la escena donde el sensor presenta valores NoData/cero, evitando introducir datos ficticios.
- **Salida:** Archivo [`data/training/dataset.tsv`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/dataset.tsv) con **14.155 observaciones** y 145 polígonos efectivos.
- **Certificación de calidad:** El módulo de pruebas [`tests/test_dataset_schema.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/tests/test_dataset_schema.py) certificó cero valores nulos, ausencia total de NaNs o infinitos, y reflectancias estrictamente acotadas en $[0.0, 1.0]$.

### 3.2 Protocolo de Validación Espacial (Zero Data Leakage)
A diferencia de los esquemas convencionales de validación aleatoria (que asignan píxeles adyacentes del mismo polígono a entrenamiento y prueba inflando artificialmente el rendimiento por autocorrelación espacial), se implementó un particionamiento por bloques espaciales mediante `Polygon_ID`:
1. **Partición de Prueba Independiente (Test Set):** 20% de los polígonos (29 polígonos, 2.708 píxeles) completamente aislados físicamente.
2. **Partición de Desarrollo (Dev Set):** 80% de los polígonos (116 polígonos, 11.447 píxeles).
3. **Validación Cruzada Espacial:** 4 folds estructurados mediante `GroupKFold` sobre los 116 polígonos de desarrollo y exportados a [`data/training/spatial_partitions.json`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/data/training/spatial_partitions.json). Todos los clasificadores consumen de forma idéntica estas particiones.

---

## 4. Modelado de Machine Learning, Optimización y Resultados Comparativos

Cada algoritmo se integró en un `Pipeline` de Scikit-Learn que combina preprocesamiento (escalado estándar `StandardScaler` cuando aplica) y el clasificador. La optimización de hiperparámetros se condujo mediante `GridSearchCV` sobre los 4 folds de desarrollo evaluando exclusivamente el **Macro-F1**, métrica imperativa dada la prevalencia heterogénea del paisaje chocoano.

### 4.1 Resumen Comparativo de Arquitecturas Evaluadas

| Pos. | Arquitectura de ML | Overall Accuracy (%) | Kappa de Cohen | Macro-F1 (Test) | CV Macro-F1 (Dev) | Tiempo Entrenamiento (s) | Hiperparámetros Óptimos |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Red Neuronal (ANN / MLP)** | 88.74% | 0.8146 | **0.7075** | **0.7103 (±0.0866)** | 103.64 s | `hidden_layer_sizes=(64, 32)`, `alpha=0.0001`, `max_iter=500` |
| 🥈 | **Support Vector Machines (SVM RBF)** | **89.70%** | **0.8283** | 0.6889 | 0.6538 (±0.0830) | 7.32 s | `kernel='rbf'`, `C=10.0`, `gamma='scale'`, `class_weight='balanced'` |
| 🥉 | **Decision Tree (CART)** | 86.41% | 0.7827 | 0.6794 | 0.6926 (±0.0923) | 7.51 s | `criterion='gini'`, `max_depth=10`, `min_samples_split=5` |
| 4️⃣ | **Regresión Logística (Baseline)** | 82.87% | 0.7343 | 0.6516 | 0.7039 (±0.0749) | 5.81 s | `penalty='l2'`, `C=1.0`, `class_weight='balanced'`, `max_iter=1000` |
| 5️⃣ | **K-Nearest Neighbors (KNN)** | 87.70% | 0.7936 | 0.6159 | 0.6805 (±0.0961) | 4.17 s | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'` |

### 4.2 Desempeño por Clase sobre el Conjunto de Prueba Independiente (Test Set)

| Clase / Cobertura | Muestras Test | ANN / MLP (F1 / Rec / Prec) | SVM RBF (F1 / Rec / Prec) | Decision Tree (F1 / Rec / Prec) | Regresión Logística (F1 / Rec / Prec) | KNN (F1 / Rec / Prec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Bosque** | 1.222 | **1.000** / 1.000 / 0.999 | 0.999 / 1.000 / 0.998 | 0.998 / 0.997 / 1.000 | 0.999 / 1.000 / 0.998 | 0.998 / 1.000 / 0.997 |
| **2. Pastos y Cultivos** | 203 | 0.409 / 0.463 / 0.366 | **0.540** / 0.507 / **0.579** | 0.490 / 0.537 / 0.450 | 0.446 / **0.709** / 0.325 | 0.428 / 0.488 / 0.381 |
| **3. Suelo Desnudo y Roca** | 1.189 | **0.888** / 0.861 / 0.918 | 0.865 / 0.852 / 0.878 | 0.835 / 0.771 / 0.910 | 0.799 / 0.684 / **0.960** | 0.841 / 0.795 / 0.893 |
| **4. Urbano y Vías** | 71 | **0.593** / 0.563 / **0.625** | 0.463 / 0.437 / 0.492 | 0.435 / 0.423 / 0.448 | 0.393 / 0.592 / 0.294 | 0.323 / 0.296 / 0.356 |
| **5. Agua** | 23 | **0.648** / **1.000** / 0.479 | 0.579 / 0.478 / **0.733** | 0.638 / 0.652 / 0.625 | 0.622 / **1.000** / 0.451 | 0.489 / 0.478 / 0.500 |

### 4.3 Discusión de Resultados y Comportamiento de las Arquitecturas
1. **Duelo en la Cumbre (ANN vs. SVM):** 
   - La **Red Neuronal Artificial (ANN / MLP)** se corona como la **campeona en Macro-F1 (0.7075 en Test y 0.7103 en CV)** gracias a su capacidad de aprender representaciones no lineales complejas para clases minoritarias desbalanceadas: logra el mejor F1 en **Urbano y Vías (0.5926)**, en **Agua (0.6479)** y en **Suelo Desnudo (0.8885)**.
   - **Support Vector Machines (SVM RBF)** logra la mayor **Exactitud Global (89.70%) y el índice Kappa más alto (0.8283)**, maximizando el margen geométrico de separación entre las coberturas predominantes (Bosque y Suelo Desnudo).
2. **Contraste entre Validación Cruzada y Prueba Independiente:** La Regresión Logística obtuvo un alto Macro-F1 en validación cruzada (0.7039), pero cayó a 0.6516 en el test independiente debido a la rigidez de su frontera hiperplanar lineal ante pendientes topográficas. Por el contrario, la ANN demostró la mayor estabilidad y generalización real (CV: 0.7103 vs Test: 0.7075, brecha inferior al 0.3%).
3. **Importancia de Bandas (Decision Tree):** El árbol de decisión identificó que la banda infrarroja de onda corta **`B11` (SWIR 1)** y la banda del borde rojo **`B05` (Red Edge 1)** concentran más del 65% de la importancia de división de nodos, ratificando la necesidad del stack de 10 bandas y justificando por qué un stack RGB básico fracasa en terrenos de alta montaña.

---

## 5. Inferencia Ráster Espacial Completa y Cuantificación de Áreas

Con el módulo [`src/predict_raster.py`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/src/predict_raster.py) se desplegaron los cinco modelos sobre la escena recortada en `EPSG:9377`. Al exigir información espectral válida en las 10 bandas de manera estricta (intersección física sin imputación sintética), el dominio analizado comprende **342.525 píxeles válidos (3.425,25 ha = 34,25 km²)**.

### Distribución Espacial de Coberturas Post-Desastre (Modelo Campeón ANN vs. SVM)

| Cobertura de la Tierra | Código | Color Oficial | Píxeles (ANN) | Área ANN (ha) | % ANN | Píxeles (SVM) | Área SVM (ha) | % SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bosque Húmedo Tropical** | 1 | `#1b7837` | 218.951 | 2.189,51 ha | 63.92 % | 218.001 | 2.180,01 ha | 63.65 % |
| **Suelo Desnudo / Deslizamientos** | 3 | `#d73027` | 117.213 | 1.172,13 ha | 34.22 % | 118.054 | 1.180,54 ha | 34.47 % |
| **Pastos y Cultivos** | 2 | `#a6d96a` | 3.822 | 38,22 ha | 1.12 % | 3.726 | 37,26 ha | 1.09 % |
| **Urbano y Vías de Comunicación** | 4 | `#e08214` | 1.993 | 19,93 ha | 0.58 % | 2.223 | 22,23 ha | 0.65 % |
| **Cuerpos de Agua (Río Ingará)** | 5 | `#2166ac` | 546 | 5,46 ha | 0.16 % | 521 | 5,21 ha | 0.15 % |
| **TOTAL VÁLIDO ANALIZADO** | — | — | **342.525** | **3.425,25 ha** | **100.00 %** | **342.525** | **3.425,25 ha** | **100.00 %** |

**Hallazgo Crítico Territorial:** Más de **1.170 hectáreas (34.2% del territorio despejado analizado)** corresponden a suelo desnudo, roca y cicatrices de deslizamientos como consecuencia directa del sismo $M_w$ 7.4. La cartografía temática generada en QGIS (`Clasificacion_Cobertura_ANN.qgz`) y exportada en [`outputs/figures/mapa_cobertura_ann.png`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/figures/mapa_cobertura_ann.png) revela cómo los movimientos en masa repotenciaron el arrastre sedimentario hacia el río Ingará e interrumpieron las vías terciarias de conexión urbana.

El archivo de simbología [`outputs/rasters/landcover_style.qml`](file:///home/nicomania11/Proyectos~/Escritorio/Emergentes/outputs/rasters/landcover_style.qml) permite renderizar de inmediato el producto en QGIS garantizando la paleta de colores canónica del proyecto.

---

## 6. Limitaciones del Estudio y Preguntas Estratégicas para las Office Hours

### 6.1 Limitaciones Reconocidas
1. **Prevalencia Desbalanceada del Paisaje:** El territorio es predominantemente selvático y agreste; coberturas antrópicas (vías, techos) y cauces fluviales representan menos del 2% del área total, lo que exige evaluar mediante métricas no sesgadas (Macro-F1) y ponderación balanceada de clases.
2. **Efectos Topográficos:** A pesar del uso de productos L2A (BOA), las pendientes abruptas del cañón del río Ingará generan variaciones de iluminación que introducen confusión entre aguas oscuras y sombras de dosel.
3. **Resolución Espacial:** Con píxeles de 10 m, senderos rurales, viviendas dispersas y cicatrices menores a 100 $\text{m}^2$ se encuentran en mezcla espectral sub-píxel.

### 6.2 Agenda y Preguntas Clave para las Office Hours (Miércoles 18 de Noviembre)
Para la sesión de asesoría con el profesor Calderón, el equipo ha preparado los siguientes cuestionamientos metodológicos:
1. *¿Es técnicamente aceptable que la partición espacial independiente (Test Set) contenga 1 polígono de Agua (23 px) y 1 de Urbano (71 px), entendiendo que esto responde al aislamiento geográfico de los polígonos para evitar fuga de datos, o recomienda enriquecer el dataset con polígonos adicionales?*
2. *¿Cómo valora metodológicamente la diferencia observada donde la Regresión Logística lidera en CV (F1 0.7039) pero SVM domina ampliamente en el Test independiente (F1 0.6889 y Kappa 0.8283)?*
3. *¿Qué criterios adicionales sugiere enfatizar al sustentar la combinación entre polígonos fotointerpretados de cicatrices reales post-evento frente a polígonos obtenidos por criterios biofísicos para coberturas naturales?*
4. *¿Qué detalles específicos de análisis espacial y de error espera encontrar en la sección de discusión del informe final y en los 15 minutos de la sustentación oral?*

---

## 7. Conclusiones Preliminares

Se consolidó con éxito un flujo de trabajo integral de GeoAI bajo software libre y Python científico:
- Se ensambló un stack de 10 bandas Sentinel-2 rigurosamente calibrado en `EPSG:9377`.
- Se implementó una separación espacial estricta sin fugas de datos agrupada por `Polygon_ID`.
- Se entrenaron, optimizaron y compararon 4 arquitecturas de ML clásicas, evidenciando la superioridad de los métodos de kernel no lineales (SVM RBF) para la cartografía geológica y de coberturas complejas.
- Se dejó implementado el pipeline de inferencia ráster y cuantificación de áreas, listo para integrar la Red Neuronal (MLP) del Rol 3 y cerrar el informe definitivo.
