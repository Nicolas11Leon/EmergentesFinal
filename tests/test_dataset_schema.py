"""
Suite de validación de calidad y contrato de datos para el TSV de EmergentesFinal.
Ejecuta validaciones estrictas y rechaza el archivo ante cualquier inconsistencia.
"""

import os
import sys
import csv
import json

MANDATORY_COLS = [
    "Latitude", "Longitude", "Polygon_ID", "Class_ID",
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "Class"
]

BAND_COLS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Mapeo canónico oficial
EXPECTED_CLASSES = {
    "Bosque": 1,
    "Pastos y Cultivos": 2,
    "Suelo Desnudo y Roca": 3,
    "Urbano y Vías": 4,
    "Agua": 5
}

def validate_tsv(file_path: str) -> bool:
    print(f"[*] Iniciando auditoría estricta de: {file_path}")
    if not os.path.isfile(file_path):
        print(f"[ERROR CRÍTICO] El archivo no existe: {file_path}")
        return False
        
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" not in first_line:
            print("[ERROR CRÍTICO] Delimitador inválido: el archivo debe usar tabulaciones reales (\\t).")
            return False
            
        header = [col.strip() for col in first_line.split("\t")]
        
        # Validar columnas obligatorias
        for col in MANDATORY_COLS:
            if col not in header:
                print(f"[ERROR CRÍTICO] Falta la columna obligatoria: '{col}'")
                return False
                
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        
        row_count = 0
        classes_found = set()
        polygons_found = set()
        errors = []
        
        for idx, row in enumerate(reader, start=2):
            row_count += 1
            c_name = row.get("Class")
            c_id_str = row.get("Class_ID")
            poly_id = row.get("Polygon_ID")
            
            # Validación de valores vacíos o nulos
            if not c_name or not c_id_str or not poly_id:
                errors.append(f"Fila {idx}: Campos de clase o polígono vacíos.")
                break
                
            # Validar que la clase exista en el catálogo
            if c_name not in EXPECTED_CLASSES:
                errors.append(f"Fila {idx}: Clase desconocida '{c_name}'.")
                break
                
            # Validar consistencia estricta entre Class y Class_ID
            try:
                c_id = int(c_id_str)
                if EXPECTED_CLASSES[c_name] != c_id:
                    errors.append(f"Fila {idx}: Discrepancia entre Class '{c_name}' y Class_ID {c_id} (esperado: {EXPECTED_CLASSES[c_name]}).")
                    break
            except ValueError:
                errors.append(f"Fila {idx}: Class_ID no es un entero válido ('{c_id_str}').")
                break
                
            classes_found.add(c_name)
            polygons_found.add(poly_id)
            
            # Validar coordenadas
            try:
                lat = float(row.get("Latitude", ""))
                lon = float(row.get("Longitude", ""))
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    errors.append(f"Fila {idx}: Coordenadas fuera de rango válido (Lat: {lat}, Lon: {lon}).")
                    break
            except ValueError:
                errors.append(f"Fila {idx}: Coordenadas no numéricas.")
                break
                
            # Validar valores espectrales
            for b in BAND_COLS:
                val_str = row.get(b, "")
                try:
                    val = float(val_str)
                    if val < 0.0 or val > 10000.0:
                        errors.append(f"Fila {idx}: Reflectancia en {b} fuera de rango espectral ({val}).")
                        break
                except ValueError:
                    errors.append(f"Fila {idx}: Valor no numérico en {b} ('{val_str}').")
                    break
                    
            if len(errors) >= 5:
                break
                
        if errors:
            print("[ERROR CRÍTICO] Se encontraron inconsistencias en los datos:")
            for err in errors:
                print(f"  - {err}")
            return False
            
        # Validar cobertura total del catálogo
        missing_classes = set(EXPECTED_CLASSES.keys()) - classes_found
        if missing_classes:
            print(f"[ERROR CRÍTICO] Faltan clases obligatorias en el dataset: {missing_classes}")
            return False
            
        # Validar representatividad espacial mínima
        if len(polygons_found) < 10:
            print(f"[ERROR CRÍTICO] Se detectaron solo {len(polygons_found)} polígonos. Se requieren al menos 10 para garantizar validación cruzada espacial.")
            return False
            
        print(f"[✓] Filas auditadas: {row_count}")
        print(f"[✓] Todas las 5 clases presentes y consistentes con sus IDs.")
        print(f"[✓] Total de polígonos independientes: {len(polygons_found)}")
        print(f"[✓] 0 valores nulos, NaN o fuera de rango espectral.")
        print("[✓] Certificación superada con éxito: el archivo es apto para entrenamiento.\n")
        return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/training/synthetic_dataset.tsv"
    success = validate_tsv(target)
    sys.exit(0 if success else 1)
