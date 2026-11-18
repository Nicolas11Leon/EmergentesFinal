# Decisiones de integración — clasificación espectral (18 de noviembre de 2026)

## Decisión A: una sola inferencia oficial

**Adoptamos el procedimiento de `src/predict_raster_hania.py` como implementación de referencia** para la ejecución final de los cinco modelos. La razón técnica es que su contrato exige simultáneamente la máscara de calidad y valores válidos en las diez bandas de Sentinel-2. Sus pruebas automatizadas se ejecutaron sin errores en Windows (suite completa: 8/8). El `src/predict_raster.py` anterior de `main` solo aplicaba máscara de calidad y B02 > 0, por lo que **no debe utilizarse para producir los resultados oficiales** mientras no sea alineado con el contrato de diez bandas.

**Contrato:** máscara oficial > 0; cada una de las diez bandas debe ser finita, positiva y distinta de NoData; misma escena, CRS, transformación afín, resolución y orden de bandas para los cinco modelos; NoData = 0; clasificaciones 1–5. A partir del ZIP oficial de Nicolás se comprobaron **471.579 píxeles en máscara inicial** y **342.525 píxeles finales**, equivalentes a **3.425,25 ha**. No imputar píxeles sin datos.

**Operación:** una sola carpeta de resultados `outputs/metrics/inferencia_10bandas/`. Se conservan resultados históricos sin sobrescribirlos. Antes de entregar, se debe anotar como obsoleto/legado el uso del script antiguo y actualizar toda documentación o comandos de ejecución (`run_pipeline.sh` incluido si corresponde) para que invoquen solamente el procedimiento oficial. No borrar código de Nicolás por iniciativa unilateral.

**Elección de modelo temático:** ANN como mapa principal debido a mayor Macro-F1 (CV 0,7103; test 0,7075) bajo la métrica priorizada. SVM sigue siendo referente alternativo por mayor Accuracy y Kappa. ANN no supera a SVM en todas las clases: empatan en Agua y Bosque, la ANN mejora el F1 de Urbano y Vías; SVM mejora ligeramente el F1 de Pastos y Suelo Desnudo. Las diferencias de rendimiento **no han sido probadas como estadísticamente significativas**.

## Decisión B: validación obligatoria píxel a píxel

**Ningún mapa definitivo se declara verificado solo porque su JSON cierre en 342.525 píxeles.** Para cada uno de los cinco GeoTIFF reales, ejecutar `audit_rasters_final.py`, que comprueba:

1. El ZIP original contiene el stack de diez bandas y la máscara, ambos alineados en EPSG:9377 y resolución de 10 m.
2. El dominio válido resultante contiene 342.525 celdas.
3. Cada GeoTIFF tiene una banda categórica, misma forma/CRS/transformación, NoData 0, valores 1–5 **únicamente** donde el dominio es válido, y 0 fuera.
4. El conteo por clase del GeoTIFF coincide exactamente con su `area_stats_*.json`; áreas en ha/km² y porcentajes coinciden salvo redondeo de presentación.
5. Los cinco resultados utilizan el mismo dominio, aunque sus predicciones por clase cambien.

### Comandos

Una vez descargado `Emergentes_Rasters_Inferencia.zip`:

```powershell
$repo = Join-Path ([Environment]::GetFolderPath('Desktop')) 'TRABAJOS JAVERIANA\Emergentes\Final Project\EmergentesFinal'
Set-Location $repo
.\.venv\Scripts\python.exe .\audit_rasters_final.py --zip "$HOME\Downloads\Emergentes_Rasters_Inferencia.zip" --source-only
```

Cuando los **cinco GeoTIFF finales estén disponibles localmente** (por ejemplo después de generarlos con la implementación oficial de Hania, tras el merge):

```powershell
.\.venv\Scripts\python.exe .\audit_rasters_final.py `
  --zip "$HOME\Downloads\Emergentes_Rasters_Inferencia.zip" `
  --rasters-dir .\outputs\rasters `
  --stats-dir .\outputs\metrics\inferencia_10bandas `
  --report .\outputs\auditorias\auditoria_final_5_modelos.json
```

Si faltan los cinco GeoTIFF locales, **la segunda auditoría debe fallar**: eso es una limitación real, no un falso positivo. El reporte se guarda bajo `outputs/auditorias/` y no en `outputs/metrics/`, porque `src/evaluate_models.py` actualmente lee todos los JSON raíz de métricas y podría incorporar erróneamente un reporte de auditoría a la tabla comparativa.

## Pendientes operativos antes de entregar

- Conseguir/verificar los cinco GeoTIFF de Hania o regenerarlos con el mismo stack y modelos, sin sobrescribir los datos originales.
- Integrar PR #3 después de confirmar que no se pierden cambios en las tablas comparativas.
- Dejar documentada la versión de scikit-learn usada para serializar LR (existe discrepancia de versiones); ejecutar con versión compatible/reentrenar el baseline si es necesario y comparar predicciones.
- Actualizar reporte e integrar los cinco algoritmos. Confirmar que el proyecto QGIS abre los archivos de entrada con rutas portables.
- Preparar los entregables que indica `final_project.pdf`: **GeoTIFF del mapa elegido, informe técnico completo y código fuente en ZIP**. Los otros cuatro mapas son anexos comparativos útiles, pero confirmar alcance en office hours.
- La presentación de máximo 15 minutos es un trabajo posterior (dos estudiantes escogidos al azar deben poder defender todo).

Esta nota es una decisión técnica del Rol 2 para proponer al equipo; no sustituye el acuerdo de revisión del PR ni constituye por sí misma una validación de los GeoTIFF que están fuera de GitHub.
