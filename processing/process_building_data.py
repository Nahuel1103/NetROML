import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_data_from_buildings import extract_data
from merge_and_reindex_rssi import merge_and_reindex

def main():
    # Calculate absolute defaults relative to this script location
    # Script is in NetROML/processing/
    # We want defaults in NetROML/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_raw_data_dir = os.path.join(project_root, "Datos_WiFi_Ceibal")
    default_output_dir = os.path.join(project_root, "buildings")
    
    parser = argparse.ArgumentParser(description="Process Wi-Fi data for a specific building.")
    parser.add_argument("--building_id", type=str, required=True, help="Building ID to process (e.g., '814', '990')")
    parser.add_argument("--raw_data_dir", type=str, default=default_raw_data_dir, 
                        help=f"Directory containing raw .tgz files (default: {default_raw_data_dir})")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, 
                        help=f"Base output directory for processed files (default: {default_output_dir})")
    parser.add_argument("--skip_extraction", action="store_true", 
                        help="Skip extraction step and only run merge/index (useful if csvs already exist)")
    parser.add_argument("--months", type=str, nargs='+', 
                        help="Specific months to process (e.g. 02 03). If not provided, process all standard months.")
    
    args = parser.parse_args()
    
    building_id = args.building_id
    raw_data_dir = args.raw_data_dir
    output_dir = args.output_dir
    mapping_file = os.path.join(script_dir, "MAC_AP_building_id.csv")
    
    print(f"=== Processing Pipeline for Building {building_id} ===")
    print(f"Raw Data Directory: {raw_data_dir}")
    print(f"Output Directory:   {output_dir}")
    print(f"Mapping File:       {mapping_file}")
    
    # 1. Extraction Step
    if not args.skip_extraction:
        print("\n--- Step 1: Extracting Data ---")
        if not os.path.exists(raw_data_dir):
            print(f"Error: Raw data directory '{raw_data_dir}' does not exist.")
            return
            
        extract_data(
            target_buildings={building_id},
            data_dir=raw_data_dir,
            mapping_file=mapping_file,
            output_base_dir=output_dir,
            subset_months=args.months
        )
    else:
        print("\n--- Step 1: Extraction Skipped ---")

    # 2. Merge and Index Step
    print("\n--- Step 2: Merging and Indexing ---")
    building_output_dir = os.path.join(output_dir, building_id)
    final_output_file = os.path.join(building_output_dir, f"building_{building_id}_all_months.csv")
    
    if not os.path.exists(building_output_dir):
        print(f"Error: Directory {building_output_dir} does not exist. Did extraction fail or was it skipped incorrectly?")
        return

    success = merge_and_reindex(
        input_dir=building_output_dir,
        output_file=final_output_file
    )
    
    if success:
        print("\n=== Pipeline Completed Successfully ===")
        print(f"Generated: {final_output_file}")
    else:
        print("\n=== Pipeline Failed at Merge Step ===")

if __name__ == "__main__":
    main()
