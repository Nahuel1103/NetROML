# =========================
# Ejemplo de uso
# =========================

from arrival_departure_model import ArrivalDepartureModel, build_client_block_counts


if __name__ == "__main__":

    csv_path = "../NetROML/buildings/990/building_990_all_months.csv"

    client_block_counts = build_client_block_counts(csv_path)

    model = ArrivalDepartureModel(
        arrival_rate=3.0,
        mean_duration=15.0,
        total_timesteps=50,
        random_seed=42,
        client_block_counts=client_block_counts
    )

    model.simulate_all_events()

    t = 10
    active = model.get_active_clients(t)

    print(f"Clientes activos en t={t}:")
    for e in active:
        block = model.get_block_index_at_timestep(e, t)
        print(f"  {e.client_id} → bloque {block}")