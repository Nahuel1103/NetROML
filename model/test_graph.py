"""
test_graph.py
=============

Tests unitarios para validar la construcción del grafo heterogéneo.

Ejecutar con:
    python test_graph.py
"""

import torch
from pathlib import Path
from network_graph_env import NetworkGraphEnv

# 0. Path Setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_root = project_root

# 1. Initialize Env
# Check if 'buildings' folder exists in project root, otherwise use root (some users put 990 in root)
if (project_root / "buildings").exists():
    data_root = project_root / "buildings"
else:
    data_root = project_root


def test_graph_structure():
    """Test básico: estructura del grafo."""
    print("\n" + "="*60)
    print("TEST 1: Estructura del Grafo")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=10,
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    # Test 1: El grafo debe tener nodos AP
    assert 'ap' in obs.x_dict, "Falta nodo tipo 'ap'"
    print(f"✓ Nodos AP presentes: {obs['ap'].x.shape[0]}")
    
    # Test 2: Edge index debe tener 2 filas
    assert obs['ap', 'connects', 'client'].edge_index.shape[0] == 2, \
        "edge_index debe tener shape [2, num_edges]"
    print(f"✓ Edge index tiene shape correcto: {obs['ap', 'connects', 'client'].edge_index.shape}")
    
    # Test 3: Índices en rango
    if obs['ap', 'connects', 'client'].edge_index.shape[1] > 0:
        max_ap = obs['ap', 'connects', 'client'].edge_index[0].max().item()
        max_client = obs['ap', 'connects', 'client'].edge_index[1].max().item()
        
        assert max_ap < env.num_aps, \
            f"Índice AP {max_ap} fuera de rango (num_aps={env.num_aps})"
        assert max_client < env.num_active_clients, \
            f"Índice Cliente {max_client} fuera de rango (num_active={env.num_active_clients})"
        
        print(f"✓ Índices en rango válido: AP[0-{env.num_aps-1}], Client[0-{env.num_active_clients-1}]")
    
    print("✅ TEST 1 PASADO\n")


def test_edge_directions():
    """Test: Verificar que las direcciones de las aristas son correctas."""
    print("\n" + "="*60)
    print("TEST 2: Direcciones de Aristas")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=10,
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    # Ejecutar varios steps para tener datos
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    
    if env.num_active_clients > 0:
        # Test downlink: AP → Cliente
        edge_idx_down = obs['ap', 'connects', 'client'].edge_index
        if edge_idx_down.shape[1] > 0:
            # Verificar que source son APs (índices 0 a num_aps-1)
            assert edge_idx_down[0].max() < env.num_aps
            # Verificar que target son Clientes (índices 0 a num_active_clients-1)
            assert edge_idx_down[1].max() < env.num_active_clients
            
            print(f"✓ Downlink: {edge_idx_down.shape[1]} aristas AP → Cliente")
            print(f"  Ejemplo: AP{edge_idx_down[0, 0].item()} → Client{edge_idx_down[1, 0].item()}")
        
        # Test uplink: Cliente → AP
        edge_idx_up = obs['client', 'connected_to', 'ap'].edge_index
        if edge_idx_up.shape[1] > 0:
            # Verificar que source son Clientes
            assert edge_idx_up[0].max() < env.num_active_clients
            # Verificar que target son APs
            assert edge_idx_up[1].max() < env.num_aps
            
            print(f"✓ Uplink: {edge_idx_up.shape[1]} aristas Cliente → AP")
            print(f"  Ejemplo: Client{edge_idx_up[0, 0].item()} → AP{edge_idx_up[1, 0].item()}")
    
    print("✅ TEST 2 PASADO\n")


def test_edge_attributes():
    """Test: Verificar que los atributos de aristas son correctos."""
    print("\n" + "="*60)
    print("TEST 3: Atributos de Aristas")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=10,
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    # Test: Edge attributes deben estar normalizados entre 0 y 1 (aprox)
    edge_attr = obs['ap', 'connects', 'client'].edge_attr
    
    if edge_attr.shape[0] > 0:
        min_attr = edge_attr.min().item()
        max_attr = edge_attr.max().item()
        
        print(f"✓ Edge attributes: min={min_attr:.3f}, max={max_attr:.3f}")
        
        # Verificar que están razonablemente normalizados
        # (normalización: (gain + 100) / 70, donde gain ∈ [-100, -30])
        # Resultado esperado: aprox [0, 1]
        assert min_attr >= -0.5, f"Edge attr demasiado bajo: {min_attr}"
        assert max_attr <= 1.5, f"Edge attr demasiado alto: {max_attr}"
        
        print(f"✓ Normalización válida")
    
    print("✅ TEST 3 PASADO\n")


def test_consistency_over_steps():
    """Test: Consistencia del grafo a lo largo de múltiples steps."""
    print("\n" + "="*60)
    print("TEST 4: Consistencia en el Tiempo")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=20,
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            break
        
        # Validaciones en cada step
        if env.num_active_clients > 0:
            # Test: Número de conexiones uplink ≤ número de clientes
            num_uplink = obs['client', 'connected_to', 'ap'].edge_index.shape[1]
            assert num_uplink <= env.num_active_clients, \
                f"Step {step}: Más conexiones uplink que clientes ({num_uplink} > {env.num_active_clients})"
            
            # Test: Clientes conectados están en client_connections
            if num_uplink > 0:
                connected_clients = obs['client', 'connected_to', 'ap'].edge_index[0].unique()
                for c_idx in connected_clients:
                    assert env.client_connections[c_idx] != -1, \
                        f"Step {step}: Cliente {c_idx} tiene arista uplink pero connection=-1"
    
    print(f"✓ Validado {step+1} steps sin errores")
    print("✅ TEST 4 PASADO\n")


def test_empty_graph():
    """Test: Comportamiento cuando no hay clientes activos."""
    print("\n" + "="*60)
    print("TEST 5: Grafo Vacío (sin clientes)")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=100,
        arrival_rate=0.1,  # Muy bajo para tener timesteps sin clientes
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    # Buscar un timestep sin clientes
    found_empty = False
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        
        if env.num_active_clients == 0:
            found_empty = True
            
            # Test: Grafos deben estar vacíos pero bien formados
            assert obs['ap', 'connects', 'client'].edge_index.shape == (2, 0), \
                "Downlink edge_index debe ser [2, 0] cuando no hay clientes"
            
            assert obs['client', 'connected_to', 'ap'].edge_index.shape == (2, 0), \
                "Uplink edge_index debe ser [2, 0] cuando no hay clientes"
            
            print(f"✓ Grafo vacío correctamente formado")
            break
        
        if done:
            obs, _ = env.reset()
    
    if found_empty:
        print("✅ TEST 5 PASADO\n")
    else:
        print("⚠️ TEST 5 OMITIDO: No se encontró timestep sin clientes\n")


def test_visualization():
    """Test visual: Imprimir ejemplo de grafo para inspección manual."""
    print("\n" + "="*60)
    print("TEST 6: Visualización Manual")
    print("="*60)
    
    env = NetworkGraphEnv(
        data_root=data_root,
        building_id=990,
        max_timesteps=10,
        random_seed=42
    )
    
    obs, _ = env.reset()
    
    # Ejecutar algunos steps para tener datos interesantes
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
    
    print(f"\nESTADO DEL SISTEMA:")
    print(f"  Timestep: {env.current_step}")
    print(f"  APs: {env.num_aps}")
    print(f"  Clientes activos: {env.num_active_clients}")
    
    if env.num_active_clients > 0:
        print(f"\nARISTAS DOWNLINK (AP → Cliente):")
        edge_idx = obs['ap', 'connects', 'client'].edge_index
        edge_attr = obs['ap', 'connects', 'client'].edge_attr
        
        for i in range(min(10, edge_idx.shape[1])):
            ap = edge_idx[0, i].item()
            client = edge_idx[1, i].item()
            gain = edge_attr[i, 0].item()
            print(f"  AP{ap:2d} → Client{client:2d} (gain_norm={gain:.3f})")
        
        if edge_idx.shape[1] > 10:
            print(f"  ... y {edge_idx.shape[1] - 10} más")
        
        print(f"\nARISTAS UPLINK (Cliente → AP, conexiones activas):")
        edge_idx_up = obs['client', 'connected_to', 'ap'].edge_index
        
        for i in range(edge_idx_up.shape[1]):
            client = edge_idx_up[0, i].item()
            ap = edge_idx_up[1, i].item()
            print(f"  Client{client:2d} → AP{ap:2d} ✓ CONECTADO")
    
    print("\n✅ TEST 6 COMPLETO (inspección manual)\n")


def run_all_tests():
    """Ejecutar todos los tests."""
    print("\n" + "="*60)
    print("EJECUTANDO SUITE DE TESTS DEL GRAFO")
    print("="*60)
    
    try:
        test_graph_structure()
        test_edge_directions()
        test_edge_attributes()
        test_consistency_over_steps()
        test_empty_graph()
        test_visualization()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ TEST FALLÓ: {e}")
        print("="*60 + "\n")
        raise
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR INESPERADO: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()