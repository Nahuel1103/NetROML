import os
import tarfile
import csv
import re
from collections import defaultdict
import glob

# Constants
TARGET_BUILDINGS = {'990'}
SCRIPT_DIR = os.path.dirname(os.path.abspath("NetROML"))
DATA_DIR = os.path.join(SCRIPT_DIR, "Datos_WiFi_Ceibal")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "MAC_AP_building_id.csv")
BUILDINGS_BASE_DIR = os.path.join(SCRIPT_DIR, "buildings_v2")

def load_mapping():
    """Loads mac_ap -> building_id mapping for target buildings."""
    mapping = {}
    with open(MAPPING_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                mac_ap, b_id = row[0].strip(), row[1].strip()
                if b_id in TARGET_BUILDINGS:
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

def process_archives():
    mapping = load_mapping()
    print(f"Loaded mapping for {len(mapping)} APs across {len(TARGET_BUILDINGS)} buildings.")
    
    # buffers: (building_id, month) -> list of rows
    # Actually, to save memory and handle large data, we can write per month for each building.
    # But since we have many tgz files, we'll open/append to the target monthly CSVs.
    
    tgz_files = sorted(glob.glob(os.path.join(DATA_DIR, "RSSI_WLCs_2018-*.tgz")))
    print(f"Found {len(tgz_files)} archive files.")
    
    # Keep track of open files to avoid repeated opening/closing
    output_files = {}

    def get_output_handle(building_id, month):
        key = (building_id, month)
        if key not in output_files:
            folder = os.path.join(BUILDINGS_BASE_DIR, building_id)
            os.makedirs(folder, exist_ok=True)
            filename = f"rssi_2018_{month}.csv"
            filepath = os.path.join(folder, filename)
            
            # If file doesn't exist, write header
            exists = os.path.exists(filepath)
            f = open(filepath, 'a', newline='')
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["mac_cliente", "mac_ap", "banda", "antena", "rssi", "date", "time"])
            output_files[key] = (f, writer)
        return output_files[key][1]

    for tgz_path in tgz_files:
        filename = os.path.basename(tgz_path)
        # Format: RSSI_WLCs_2018-MM-DD_HH_MM.tgz
        match = re.search(r"2018-(\d{2})-\d{2}_(\d{2}_\d{2})", filename)
        if not match:
            continue
            
        month = match.group(1)
        if month not in ['02', '03']:
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
                                    if building_id:
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

if __name__ == "__main__":
    process_archives()
