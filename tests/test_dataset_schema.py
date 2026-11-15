"""
Suite de validación de calidad y contrato de datos para el TSV de EmergentesFinal.
Ejecutable sin dependencias externas usando únicamente la biblioteca estándar.
"""

import os
import sys
import csv

MANDATORY_COLS = [
    "Latitude", "Longitude", "Polygon_ID", "Class_ID",
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "Class"
]

EXPECTED_CLASSES = {"Bosque", "Pastos y Cultivos", "Suelo Desnudo y Roca", "Urbano y Vías", "Agua"}
BAND_COLS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

def validate_tsv(file_path: str) -> bool:
    print(f"[*] Validando archivo: {file_path}")
    if not os.path.isfile(file_path):
        print(f"[ERROR] El archivo no existe: {file_path}")
        return False
        
    with open(file_path, "r", encoding="utf-8") as f:
        # Detectar delimitador
        first_line = f.readline()
        if "\t" not in first_line:
            print("[ERROR] El archivo no utiliza tabulaciones reales (\\t) como separador.")
            return False
            
        header = [col.strip() for col in first_line.split("\t")]
        
        # Validar columnas
        for col in MANDATORY_COLS:
            if col not in header:
                print(f"[ERROR] Falta la columna obligatoria: '{col}'")
                return False
                
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        
        row_count = 0
        classes_found = set()
        polygons_found = set()
        invalid_values = 0
        
        for idx, row in enumerate(reader, start=2):
            row_count += 1
            c = row.get("Class")
            p = row.get("Polygon_ID")
            classes_found.add(c)
            polygons_found.add(p)
            
            # Validar bandas
            for b in BAND_COLS:
                val_str = row.get(b, "")
                try:
                    val = float(val_str)
                    if val < 0.0 or val > 10000.0:  # Acepta [0, 1] reflectancia o [0, 10000] DN
                        invalid_values += 1
                        if invalid_values <= 3:
                            print(f"[WARN] Fila {idx}: Valor fuera de rango en {b} ({val})")
                except ValueError:
                    print(f"[ERROR] Fila {idx}: Valor no numérico en {b} ('{val_str}')")
                    return False
                    
        print(f"[✓] Filas totales analizadas: {row_count}")
        print(f"[✓] Clases detectadas ({len(classes_found)}): {sorted(list(classes_found))}")
        print(f"[✓] Polígonos únicos detectados: {len(polygons_found)}")
        
        if not EXPECTED_CLASSES.issubset(classes_found):
            missing = EXPECTED_CLASSES - classes_found
            print(f"[WARN] Faltan clases del catálogo oficial: {missing}")
            
        if len(polygons_found) < 10:
            print(f"[WARN] Pocos polígonos detectados ({len(polygons_found)}). Se recomiendan >= 15 para GroupKFold.")
            
        print("[✓] Control de calidad de esquema superado con éxito.\n")
        return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/training/synthetic_dataset.tsv"
    success = validate_tsv(target)
    sys.exit(0 if success else 1)
