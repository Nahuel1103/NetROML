import pandas as pd

# Define the target building IDs
target_buildings = {84, 814, 839, 842, 990, 1361}

# Read the CSV file
df = pd.read_csv('from_mac_to_building_id.csv')

# Filter for the target buildings
filtered_df = df[df['building_id'].isin(target_buildings)]

# Select relevant columns
filtered_df = filtered_df[['MAC_AP', 'building_id']]

# Sort by building_id and MAC_AP
filtered_df = filtered_df.sort_values(by=['building_id', 'MAC_AP'])

# Save to CSV
output_file = 'MAC_AP_building_id.csv'
filtered_df.to_csv(output_file, index=False)
print(f"Exported to {output_file}")
