# Cuaderno de Estudio y Libreto de Sustentación (GeoAI)
**Proyecto Final: Clasificación de Coberturas Terrestres Post-Sismo (San José del Palmar, Chocó)**

---

## 1. Libreto del Discurso para la Sustentación (Paso a Paso)

### Introducción y Justificación (De QGIS a GeoAI)
> *"Buenos días, profesor. Somos el equipo conformado por Nicolás León, Andrés Felipe Barrios, Hania Valentina Carreño y Jorge Sierra.*
>
> *En los laboratorios pasados trabajábamos la clasificación supervisada directamente dentro de la interfaz gráfica de QGIS. Sin embargo, para este proyecto final dimos el salto metodológico a **GeoAI programático en Python**.*
>
> *¿Por qué migrar a GeoAI?*
> 1. ***Escalabilidad:*** *En QGIS todo depende de clics manuales y memoria de escritorio; en Python podemos procesar millones de píxeles y escenas satelitales completas de forma automatizada.*
> 2. ***Precisión algorítmica y control:*** *Podemos optimizar hiperparámetros finos (`GridSearchCV`) y calibrar las fronteras matemáticas de decisión.*
> 3. ***Rigor científico y reproducibilidad:*** *Todo el flujo queda en código auditable, con semillas fijas y control estricto de particiones para garantizar cero fuga de información."*

---

### El Problema y la Regla de Oro (Cero Fuga Espacial)
> *"El caso de estudio es el sismo de magnitud $M_w$ 7.4 del 10 de agosto de 2026 en San José del Palmar (Chocó). En lugar de arriesgar vidas caminando la selva montañosa tras el desastre, tomamos una imagen Sentinel-2 del 18 de agosto con 10 bandas espectrales.*
>
> *Nuestro principio técnico no negociable fue garantizar **cero fuga espacial (*Zero Spatial Data Leakage*)**:*
> * *Si hacíamos una separación aleatoria por píxeles sueltos, el modelo memorizaba píxeles vecinos idénticos del mismo potrero o derrumbe (Primera Ley de Tobler) y sacaba una nota alta falsa.*
> * *Nosotros separamos por **polígonos completos (`Polygon_ID`)**: 116 polígonos de desarrollo para entrenar con 4 folds espaciales, y 29 polígonos de prueba (2.708 píxeles) totalmente aislados bajo llave para el examen final."*

---

### La Competencia de Modelos y el Criterio Macro-F1
> *"Pusimos a competir cinco arquitecturas de aprendizaje automático sobre las mismas 10 bandas y con los mismos datos:*
> 1. *Regresión Logística (baseline lineal).*
> 2. *Árbol de Decisión (CART).*
> 3. *K-Nearest Neighbors (KNN).*
> 4. *Support Vector Machines (SVM con kernel RBF).*
> 5. *Red Neuronal Artificial (ANN / MLP).*
>
> *Evaluamos el desempeño usando **Macro-F1**, que promedia el éxito en las 5 coberturas dándoles el mismo peso a todas. Esto es vital porque el paisaje es muy desbalanceado: el bosque y las rocas ocupan el 98 %, mientras que el río y las vías son menos del 2 %. Un modelo que solo acierte en bosque tendría alto Accuracy pero sería inútil para la emergencia.*
>
> *El modelo ganador en equilibrio fue la **Red Neuronal Artificial (ANN / MLP)** con un Macro-F1 de **0,7075**, destacándose por su alta precisión en la detección de vías y pueblo (62,5 %), superando a SVM que tendía a confundirlos."*

---

### La Conclusión Territorial y el Mapa Final
> *"Con la Red Neuronal seleccionada, realizamos la inferencia ráster sobre los **342.525 píxeles válidos (3.425,25 hectáreas)** del cañón del río Ingará y exportamos el GeoTIFF a QGIS con su simbología oficial.*
>
> *La conclusión territorial es contundente:*
> * *El **34,2 % del cañón (1.172,13 hectáreas)** quedó expuesto con suelo desnudo, roca y cicatrices de deslizamientos como impacto directo del sismo.*
> * *Esto evidencia la magnitud del desastre: laderas desprendidas que taponaron afluentes hacia el río Ingará y cortaron las vías terciarias que comunican a San José del Palmar.*
> * *El bosque húmedo remanente conservó el 63,9 % (2.189,51 ha), los pastos ocuparon el 1,1 % (38,22 ha) y el casco urbano el 0,6 % (19,93 ha)."*

---

## 2. Podio Comparativo de los 5 Modelos

| Puesto | Macro-F1 (Equilibrio) | Accuracy (Aciertos) | Tiempo HPO (Velocidad) | Rol y Lección Técnica |
| :---: | :--- | :--- | :--- | :--- |
| 🥇 **1º** | **ANN / MLP** (0,7075) | **SVM RBF** (89,70 %) | **KNN** (4,17 s) | **ANN:** Modelo campeón por equilibrio; no sacrificó clases pequeñas. Pintó el mapa final. |
| 🥈 **2º** | **SVM RBF** (0,6889) | **ANN / MLP** (88,74 %) | **Reg. Logística** (5,81 s) | **SVM:** Fronteras curvas limpias para bosque vs. roca masiva. |
| 🥉 **3º** | **Árbol CART** (0,6794) | **KNN** (87,70 %) | **SVM RBF** (7,32 s) | **Árbol:** Explicó que `B08` (NIR, 33,9 %) y `B04` (Rojo, 32,6 %) aportan el 66,5 % de la información. |
| **4º** | **Reg. Logística** (0,6516) | **Árbol CART** (86,41 %) | **Árbol CART** (7,51 s) | **Logística:** Baseline lineal obligatorio; confundió suelo con pasturas en relieve agreste. |
| **5º** | **KNN** (0,6159) | **Reg. Logística** (82,87 %) | **ANN / MLP** (103,64 s) | **KNN:** Muy veloz, pero la dispersión en 10 bandas perjudicó las clases minoritarias. |

---

## 3. Las 10 Bandas de Sentinel-2 (La Firma Espectral)
1. `B02` (Azul - 490 nm): Agua y techos/vías.
2. `B03` (Verde - 560 nm): Pico verde de la clorofila.
3. `B04` (Rojo - 665 nm): Absorción foliar; clave para separar suelo de vegetación.
4. `B05` (Red Edge 1 - 705 nm): Rampa del borde rojo; sensibilidad a estrés y tipo de follaje.
5. `B06` (Red Edge 2 - 740 nm): Transición foliar; separa pastos de dosel arbóreo denso.
6. `B07` (Red Edge 3 - 783 nm): Hombro del infrarrojo; biomasa.
7. `B08` (NIR amplio - 842 nm): Meseta celular de vegetación sana; nula en agua.
8. `B8A` (NIR estrecho - 865 nm): Infrarrojo libre de interferencia de vapor de agua.
9. `B11` (SWIR 1 - 1610 nm): Humedad y respuesta mineral de roca/cicatrices de derrumbes.
10. `B12` (SWIR 2 - 2190 nm): Minerales arcillosos y materiales urbanos.

---

## 4. Resumen de Librerías y Herramientas Utilizadas
* `rasterio`: Lectura y escritura de rásteres satelitales en `EPSG:9377`.
* `geopandas` / `shapely`: Intersección de polígonos de verdad de terreno (`ground_truth.gpkg`).
* `pandas` / `numpy`: Manipulación de la matriz de datos `dataset.tsv` (14.155 filas).
* `scikit-learn`: Entrenamiento de clasificadores, `GridSearchCV` en 4 folds espaciales y métricas.
* `joblib`: Serialización de modelos (`.joblib`).
* `QGIS`: Maquetación cartográfica final con escala métrica, norte y leyenda validada.
