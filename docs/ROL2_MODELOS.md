# Rol 2 - Clasificacion espectral con modelos clasicos

## Objetivo

Entrenar y evaluar Decision Tree, SVM (RBF) y KNN
sobre el mismo dataset y las mismas particiones
espaciales utilizadas por el baseline de Regresion
Logistica del proyecto.

## Datos y metodologia

- Dataset: data/training/dataset.tsv
- Observaciones: 14.155 pixeles
- Variables predictoras: B02, B03, B04, B05,
  B06, B07, B08, B8A, B11 y B12
- Variable objetivo: Class
- Agrupacion espacial: Polygon_ID
- Desarrollo: 11.447 observaciones
- Prueba independiente: 2.708 observaciones
- Validacion cruzada: 4 folds exactos de
  data/training/spatial_partitions.json
- Optimizacion: GridSearchCV con Macro-F1

Los poligonos de prueba nunca participan en la
busqueda de hiperparametros. SVM y KNN utilizan
StandardScaler dentro del Pipeline.

## Reproduccion

Ejecutar desde la raiz del repositorio con las
dependencias instaladas:

    python src/model_classical.py --model all --check-only
    python -m unittest discover -s tests -p test_model_classical.py -v

Para entrenar individualmente:

    python src/model_classical.py --model decision_tree --jobs 2
    python src/model_classical.py --model svm --jobs 2
    python src/model_classical.py --model knn --jobs 2

## Resultados

| Modelo | CV Macro-F1 | Test Macro-F1 | Accuracy | Kappa |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7039 | 0.6516 | 0.8287 | 0.7343 |
| Decision Tree | 0.6926 | 0.6794 | 0.8641 | 0.7827 |
| SVM | 0.6538 | 0.6889 | 0.8970 | 0.8283 |
| KNN | 0.6805 | 0.6159 | 0.8770 | 0.7936 |

Los mejores hiperparametros de cada modelo se
encuentran en outputs/metrics/*.json.

## Trazabilidad y artefactos

Los modelos serializados se almacenan en models/.
Las metricas se almacenan en outputs/metrics/.
Las matrices de confusion y la importancia de
bandas del arbol se encuentran en outputs/figures/.

Los archivos *_provenance.json contienen las
huellas SHA-256 del TSV y de las particiones,
el orden de bandas y la semilla utilizada.

## Limitaciones

La prueba utiliza poligonos espacialmente separados,
pero las observaciones de un mismo poligono no son
independientes entre si.

Agua y Urbano y Vias cuentan con un solo poligono
cada una en prueba, por lo que sus metricas tienen
representatividad espacial limitada.

Las etiquetas combinan fotointerpretacion y
criterios espectrales, y no constituyen por si
solas una validacion de campo independiente.

SVM obtuvo el mayor Macro-F1 observado en el test
entre estos cuatro modelos, mientras que la
Regresion Logistica obtuvo el mayor promedio de
Macro-F1 de validacion cruzada. No se deben usar
los resultados del test para reajustar el HPO.

La comparacion final de los cinco modelos queda
pendiente de incorporar la ANN del Rol 3.
