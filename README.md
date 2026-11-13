# EmergentesFinal

## Final Project: A Comparative Analysis of Machine Learning Architectures for Satellite-Based Land Cover Classification

### Descripción
Proyecto comparativo de modelos de Machine Learning aplicados a la clasificación supervisada de cobertura terrestre usando imágenes satelitales multiespectrales (Sentinel-2).

### Arquitecturas a Evaluar
1. **Regresión Logística (LR)** - Baseline
2. **Árboles de Decisión (DT)** / Ensembles (Random Forest, Boosting)
3. **Máquinas de Vectores de Soporte (SVM)**
4. **Redes Neuronales Artificiales (ANN)**
5. **K-Nearest Neighbors (KNN)** o **Naive Bayes (NB)**

### Fases del Proyecto
- **Fase 1: Definición del Problema (Semana 1)**: Selección del evento/ROI, límites del área de estudio, metadatos y extracción tabular (TSV).
- **Fase 2: Preprocesamiento y Entrenamiento (Semana 2)**: Validación de dataset TSV, optimización de hiperparámetros (HPO), matriz de confusión, métricas (Accuracy, Kappa, Precision, Recall, F1).
- **Fase 3: Síntesis e Inferencia Raster (Semana 3)**: Inferencia completa de la escena, exportación a GeoTIFF georreferenciado, visualización temática y cuantificación de áreas.
