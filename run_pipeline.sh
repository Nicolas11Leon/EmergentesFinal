#!/usr/bin/env bash
# ==============================================================================
# Pipeline Maestro de Ejecución End-to-End: Rol 1 (Data Engineering & Baseline)
# Proyecto: EmergentesFinal - Clasificación Multiespectral Sentinel-2
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "    EMERGENTES FINAL - PIPELINE DE DATOS Y MODELO 1 (BASELINE)       "
echo "======================================================================"

# 1. Verificar entorno virtual
PYTHON_BIN=".venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "[*] Entorno virtual no encontrado en .venv. Usando python3 del sistema..."
    PYTHON_BIN="python3"
fi

echo "[*] Intérprete Python: $PYTHON_BIN"

# 2. Paso 0: Consolidar verdad de terreno vectorial si no existe
if [ ! -f "data/vectors/ground_truth.gpkg" ]; then
    echo "[*] Generando capa consolidada de verdad de terreno..."
    $PYTHON_BIN src/create_ground_truth.py
else
    echo "[✓] Capa vectorial encontrada: data/vectors/ground_truth.gpkg"
fi

# 3. Paso 1: Ensamblaje del Stack de 10 Bandas
echo ""
echo "--- PASO 1: Ensamblaje y Alineación del Stack Multiespectral ---"
if $PYTHON_BIN src/01_prepare_stack.py; then
    echo "[✓] Paso 1 completado exitosamente."
else
    echo "[!] Paso 1 no pudo completarse (faltan bandas B05, B06, B07, B8A en data/raw/)."
    echo "    Cuando coloques los archivos TIFF correspondientes en data/raw/, vuelve a ejecutar este script."
    exit 1
fi

# 4. Paso 2: Extracción Tabular a TSV
echo ""
echo "--- PASO 2: Extracción Espacial Tabular a dataset.tsv ---"
$PYTHON_BIN src/02_extract_tsv.py

# 5. Paso 3: Validación Estricta de Calidad y Esquema de Datos
echo ""
echo "--- PASO 3: Validación Estricta con test_dataset_schema.py ---"
$PYTHON_BIN tests/test_dataset_schema.py data/training/dataset.tsv

# 6. Paso 4: Generación de Particiones Espaciales Compartidas (Zero Data Leakage)
echo ""
echo "--- PASO 4: Generación de Particiones Espaciales (GroupKFold por Polygon_ID) ---"
$PYTHON_BIN src/spatial_split.py

# 7. Paso 5: Entrenamiento y HPO del Modelo 1 (Regresión Logística Baseline)
echo ""
echo "--- PASO 5: Entrenamiento y HPO del Modelo 1 (Logistic Regression) ---"
$PYTHON_BIN src/model_logistic_regression.py data/training/dataset.tsv

echo ""
echo "======================================================================"
echo "[✓] PIPELINE EJECUTADO CON ÉXITO AL 100%!"
echo "    - Stack 10 bandas : data/processed/stack_10bands.tif"
echo "    - Dataset tabular : data/training/dataset.tsv"
echo "    - Particiones     : data/training/spatial_partitions.json"
echo "    - Modelo guardado : models/logistic_regression.joblib"
echo "    - Métricas JSON   : outputs/metrics/logistic_regression.json"
echo "======================================================================"
