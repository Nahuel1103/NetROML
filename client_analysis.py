import pandas as pd
import os

def load_client_data(filepath):
    """
    Loads Wi-Fi RSSI data from a CSV file.
    Expected columns: mac_cliente, mac_ap, banda, antena, rssi
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def get_client_frequencies(df):
    """
    Calculates the distribution of connections per client.
    Returns a Series: mac_cliente -> count
    """
    return df['mac_cliente'].value_counts()

def categorize_clients(freq_series, threshold):
    """
    Categorizes clients based on a frequency threshold.
    Returns two lists: frequent_clients, sporadic_clients
    """
    frequent = freq_series[freq_series >= threshold].index.tolist()
    sporadic = freq_series[freq_series < threshold].index.tolist()
    return frequent, sporadic

def filter_by_clients(df, client_list):
    """
    Filters the DataFrame to include only the specified clients.
    """
    return df[df['mac_cliente'].isin(client_list)]

def sort_records_by_connection(df):
    """
    Sorts the DataFrame by mac_cliente and mac_ap to group measurements.
    """
    return df.sort_values(by=['mac_cliente', 'mac_ap']).reset_index(drop=True)

def get_frequency_stats(freq_series):
    """
    Returns basic statistics of client frequencies.
    """
    return freq_series.describe()
