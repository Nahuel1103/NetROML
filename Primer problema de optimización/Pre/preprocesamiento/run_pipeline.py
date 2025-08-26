import os
import subprocess
from pathlib import Path

# --- Configuración (¡EDITA ESTO!) ---
BUILDING_ID = 990                  # Ejemplo: Edificio 990
B5G = True                         # Banda 5GHz (True) o 2.4GHz (False)
TRAIN_MONTHS = ["marzo", "abril", "mayo", "junio", "julio", "agosto"]
VAL_MONTHS = ["setiembre", "octubre", "noviembre", "diciembre"]

# --- Paths automáticos (no tocar) ---
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / "datos_ceibal"
GRAPHS_DIR = SCRIPTS_DIR.parent / "graphs"

def run_script(script_name, args=None):
    """Ejecuta un script de la carpeta scripts/ con argumentos."""
    command = ["python", f"{SCRIPTS_DIR}/{script_name}"]
    if args:
        command.extend(args)
    subprocess.run(command, check=True)

def main():
    print("=== INICIANDO PIPELINE ===")

    # --- Paso 1: Mapeo MAC → building_id ---
    print("\n[PASO 1] Generando mapeo MAC a building_id...")
    run_script("create_mac_hexa_buildingid_df.py")

    # --- Paso 2: Procesar meses (train + val) ---
    print("\n[PASO 2] Procesando meses individuales...")
    for month in TRAIN_MONTHS + VAL_MONTHS:
        script = "process_august_ceibal_data.py" if month == "agosto" else "process_ceibal_data.py"
        print(f"  - Procesando {month}...")
        run_script(script, [
            "--building_id", str(BUILDING_ID),
            "--b5g", str(B5G)
        ])

    # --- Paso 3: Unir en train/val ---
    print("\n[PASO 3] Combinando meses en train/val...")
    run_script("join_graphs.py", [
        "--building_id", str(BUILDING_ID),
        "--b5g", str(B5G),
        "--train_months", ",".join(TRAIN_MONTHS),
        "--val_months", ",".join(VAL_MONTHS)
    ])

    # --- Paso 4 (Opcional): Coeficientes ---
    print("\n[PASO 4] Calculando coeficientes para validación...")
    run_script("channel_coeffs.py", [
        "--input", f"{GRAPHS_DIR}/{'5' if B5G else '2_4'}_{BUILDING_ID}/val_{'5' if B5G else '2_4'}_graphs_{BUILDING_ID}.pkl"
    ])

    print("\n=== PIPELINE COMPLETADO ===")
    print(f"Resultados en: {GRAPHS_DIR}/")

if __name__ == "__main__":
    main()