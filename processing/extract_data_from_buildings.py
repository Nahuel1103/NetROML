import os
import tarfile
import csv
import re
import glob

# Constants
DEFAULT_TARGET_BUILDINGS = {'814'}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming NetROML is the parent or current dir, adjusting to be relative to this script
# Original logic: os.path.dirname(os.path.abspath("NetROML")) -> depends on CWD
# We will use relative paths from this script location for defaults
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Datos_WiFi_Ceibal")
DEFAULT_MAPPING_FILE = os.path.join(SCRIPT_DIR, "MAC_AP_building_id.csv")
DEFAULT_BUILDINGS_BASE_DIR = os.path.join(PROJECT_ROOT, "buildings")

def load_mapping(mapping_file, target_buildings=None):
    """Loads mac_ap -> building_id mapping for target buildings."""
    mapping = {}
    if not os.path.exists(mapping_file):
        print(f"Warning: Mapping file not found at {mapping_file}")
        return mapping
        
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return mapping
            
        for row in reader:
            if len(row) >= 2:
                mac_ap, b_id = row[0].strip(), row[1].strip()
                if target_buildings is None or b_id in target_buildings:
                    mapping[mac_ap] = b_id
    return mapping

def parse_line(line):
    """Parses a single line from datos_RSSI_WLCX.txt.
    Format: SNMPv2-SMI::enterprises.14179.2.1.11.1.5.CLIENT_MAC.AP_MAC.BAND.ANTENNA = INTEGER: RSSI
    """
    try:
        if " = INTEGER: " not in line:
            return None
        
        oid_part, rssi_val = line.split(" = INTEGER: ")
        rssi = rssi_val.strip()
        
        # OID prefix: SNMPv2-SMI::enterprises.14179.2.1.11.1.5.
        # We need to extract the numbers after the last dot of the prefix
        prefix = "SNMPv2-SMI::enterprises.14179.2.1.11.1.5."
        if not oid_part.startswith(prefix):
            return None
            
        remaining = oid_part[len(prefix):]
        parts = remaining.split('.')
        
        if len(parts) < 14:
            return None
            
        mac_cliente = ".".join(parts[0:6])
        mac_ap = ".".join(parts[6:12])
        banda = parts[12]
        antena = parts[13]
        
        return {
            "mac_cliente": mac_cliente,
            "mac_ap": mac_ap,
            "banda": banda,
            "antena": antena,
            "rssi": rssi
        }
    except Exception:
        return None

def extract_data(target_buildings, data_dir=DEFAULT_DATA_DIR, mapping_file=DEFAULT_MAPPING_FILE, output_base_dir=DEFAULT_BUILDINGS_BASE_DIR, subset_months=None):
    """
    Extracts RSSI data for specific buildings from .tgz archives.
    
    Args:
        target_buildings (set or list): Set of building IDs to extract data for.
        data_dir (str): Directory containing RSSI_WLCs_*.tgz files.
        mapping_file (str): Path to CSV mapping MAC AP to building ID.
        output_base_dir (str): Directory where building subdirectories will be created.
        subset_months (list): Optional list of months (e.g., ['02', '03']) to process. If None, default set is used.
    """
    if isinstance(target_buildings, str):
        target_buildings = {target_buildings}
    else:
        target_buildings = set(target_buildings)
        
    mapping = load_mapping(mapping_file, target_buildings)
    print(f"Loaded mapping for {len(mapping)} APs across {len(target_buildings)} buildings.")
    
    if not mapping:
        print("No mapping found for target buildings. Aborting.")
        return

    tgz_files = sorted(glob.glob(os.path.join(data_dir, "RSSI_WLCs_2018-*.tgz")))
    print(f"Found {len(tgz_files)} archive files in {data_dir}.")
    
    # Keep track of open files to avoid repeated opening/closing
    output_files = {}

    def get_output_handle(building_id, month):
        key = (building_id, month)
        if key not in output_files:
            folder = os.path.join(output_base_dir, building_id)
            os.makedirs(folder, exist_ok=True)
            filename = f"rssi_2018_{month}.csv"
            filepath = os.path.join(folder, filename)
            
            # If file doesn't exist, write header
            # Note: We append by default here, but cleaner might be to check if we are starting fresh in a full run
            # For now, we stick to append to match original behavior of processing multiple archives
            exists = os.path.exists(filepath)
            f = open(filepath, 'a', newline='')
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["mac_cliente", "mac_ap", "banda", "antena", "rssi", "date", "time"])
            output_files[key] = (f, writer)
        return output_files[key][1]

    valid_months = subset_months if subset_months else ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    for tgz_path in tgz_files:
        filename = os.path.basename(tgz_path)
        # Format: RSSI_WLCs_2018-MM-DD_HH_MM.tgz
        match = re.search(r"2018-(\d{2})-\d{2}_(\d{2}_\d{2})", filename)
        if not match:
            continue
            
        month = match.group(1)
        if month not in valid_months:
            continue

        time_str = match.group(2).replace('_', ':')
        date_str = filename.split('_')[2] # 2018-MM-DD
        
        print(f"Processing {filename}...")
        
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                # We are interested in RSSI_WLCs/datos_RSSI_WLC1.txt and RSSI_WLCs/datos_RSSI_WLC3.txt
                for member in tar.getmembers():
                    if "datos_RSSI_WLC" in member.name and member.name.endswith(".txt"):
                        f = tar.extractfile(member)
                        if f:
                            for line_bytes in f:
                                line = line_bytes.decode('utf-8', errors='ignore').strip()
                                parsed = parse_line(line)
                                if parsed:
                                    building_id = mapping.get(parsed['mac_ap'])
                                    if building_id and building_id in target_buildings:
                                        writer = get_output_handle(building_id, month)
                                        writer.writerow([
                                            parsed['mac_cliente'],
                                            parsed['mac_ap'],
                                            parsed['banda'],
                                            parsed['antena'],
                                            parsed['rssi'],
                                            date_str,
                                            time_str
                                        ])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Close all handles
    for f, writer in output_files.values():
        f.close()
    print("Completed processing.")

def main():
    # Maintain backward compatibility behavior
    extract_data(DEFAULT_TARGET_BUILDINGS)

if __name__ == "__main__":
    main()
