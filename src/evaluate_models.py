"""
Módulo de Evaluación Comparativa de Modelos (Summary Table & Benchmark).
Lee los resultados JSON de los 5 modelos en outputs/metrics/, compila la tabla
comparativa oficial y genera el código en Markdown y LaTeX para el informe final.
"""

import os
import sys
import json
import glob

def compile_model_comparison(metrics_dir: str = "outputs/metrics"):
    print("[*] Compilando evaluación comparativa de arquitecturas ML...")
    json_files = glob.glob(os.path.join(metrics_dir, "*.json"))
    
    # Filtrar solo archivos de métricas de modelos (excluir estadísticas de área)
    model_files = [f for f in json_files if not os.path.basename(f).startswith("area_stats")]
    
    if not model_files:
        print("[!] No se encontraron archivos de métricas en outputs/metrics/.")
        return False
        
    models_data = []
    for mf in sorted(model_files):
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
            models_data.append(data)
            
    # Ordenar por Macro-F1 descendente
    models_data.sort(key=lambda x: x.get("test_macro_f1", 0), reverse=True)
    
    print("\n" + "=" * 90)
    print("                 TABLA COMPARATIVA OFICIAL DE RENDIMIENTO DE MODELOS")
    print("=" * 90)
    header = f"{'Modelo':<30} | {'OA (%)':<8} | {'Kappa':<7} | {'Macro-F1':<8} | {'CV F1 (±std)':<14} | {'Tiempo (s)':<10}"
    print(header)
    print("-" * 90)
    
    md_lines = [
        "# Tabla Comparativa de Arquitecturas de Machine Learning",
        "",
        "| Modelo | Overall Accuracy | Kappa de Cohen | Macro-F1 | CV F1 (Dev) | Tiempo Entrenamiento |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    tex_rows = []
    
    for m in models_data:
        name = m.get("model_name", "Desconocido")
        oa = f"{m.get('test_overall_accuracy', 0)*100:.2f}%"
        kappa = f"{m.get('test_kappa', 0):.4f}"
        macro_f1 = f"{m.get('test_macro_f1', 0):.4f}"
        cv_f1 = f"{m.get('cv_macro_f1_mean', 0):.4f} (±{m.get('cv_macro_f1_std', 0):.4f})"
        t = f"{m.get('training_time_seconds', 0):.2f} s"
        
        row_str = f"{name:<30} | {oa:<8} | {kappa:<7} | {macro_f1:<8} | {cv_f1:<14} | {t:<10}"
        print(row_str)
        
        md_lines.append(f"| **{name}** | {oa} | {kappa} | **{macro_f1}** | {cv_f1} | {t} |")
        tex_rows.append(f"        \\textbf{{{name}}} & {oa} & {kappa} & \\textbf{{{macro_f1}}} & {cv_f1} & {t} \\\\")
        
    print("=" * 90)
    
    # Exportar Markdown
    md_path = os.path.join(metrics_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[✓] Tabla Markdown exportada a: {md_path}")
    
    # Exportar fragmento LaTeX listo para report.tex
    tex_content = """\\begin{table}[htbp]
    \\centering
    \\caption{Comparación de rendimiento entre arquitecturas de Machine Learning sobre conjunto de prueba espacial independiente.}
    \\label{tab:model_comparison}
    \\small
    \\begin{tabular}{lrrrrr}
        \\toprule
        \\textbf{Modelo} & \\textbf{Overall Acc.} & \\textbf{Kappa} & \\textbf{Macro-F1} & \\textbf{CV F1 (Dev)} & \\textbf{Tiempo} \\\\
        \\midrule
""" + "\n".join(tex_rows) + """
        \\bottomrule
    \\end{tabular}
\\end{table}
"""
    tex_path = os.path.join(metrics_dir, "comparison_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[✓] Tabla LaTeX exportada a: {tex_path}")
    return True

if __name__ == "__main__":
    compile_model_comparison()
