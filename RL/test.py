import torch
import numpy as np
from network_env import NetworkEnvironment
from stable_baselines3 import PPO
from utils import load_channel_matrix

def decode_action(action_code, num_power_levels):
    """Traduce el número entero de la acción a texto legible"""
    if action_code == 0:
        return "🔴 APAGADO"
    
    act_idx = action_code - 1
    ch = act_idx // num_power_levels
    pw_level = act_idx % num_power_levels
    return f"🟢 Canal {ch} | Potencia Nivel {pw_level}"

def test_allocation_logic():
    # 1. Cargar entorno y modelo
    num_links = 5
    num_channels = 3
    num_power_levels = 2
    
    # Cargar matriz (modo sintético para test rápido)
    channel_iter = load_channel_matrix(990, False, num_links, synthetic=False)
    
    env = NetworkEnvironment(
        num_links=num_links, 
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        channel_matrix_iter=channel_iter
    )
    
    # Cargar modelo (o usar random si no tienes uno entrenado aún)
    try:
        model = PPO.load("./models/gnn_ppo_network_final")
        print("✓ Modelo cargado.")
    except:
        print("⚠️ No se encontró modelo, usando acciones aleatorias para probar visualización.")
        model = None

    obs, _ = env.reset()
    
    print("\n🔍 --- INICIO DE DEPURACIÓN DE ACCIONES ---")
    
    # Correr 5 pasos
    for i in range(5):
        if model:
            # Truco para usar el forward de la policy manualmente
            with torch.no_grad():
                obs_tensor = {
                    'channel_matrix': torch.FloatTensor(obs['channel_matrix']).unsqueeze(0),
                    'mu_k': torch.FloatTensor(obs['mu_k']).unsqueeze(0)
                }
                actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
                action = actions.cpu().numpy()[0]
        else:
            action = env.action_space.sample()
            
        obs, reward, _, _, info = env.step(action)
        
        print(f"\n⏱️  STEP {i+1}")
        print(f"   Reward: {reward:.4f} | Rate Total: {info['rate']:.2f}")
        
        # AQUÍ ESTÁ LA MAGIA: Verificación visual de la asignación
        print(f"   {'Enlace':<8} {'Acción Raw':<12} {'Decisión Decodificada'}")
        print(f"   {'-'*50}")
        
        allocations = info['allocation_desc'] # Viene del cambio en el Paso A
        
        for link_id, (ch, p_val) in enumerate(allocations):
            raw_act = action[link_id]
            human_readable = decode_action(raw_act, num_power_levels)
            
            # Validación visual: ¿Coincide la lógica?
            print(f"   Link {link_id:<3} {raw_act:<12} {human_readable} (P={p_val:.4f})")
            
        # Verificar matriz Phi
        phi = info['action_phi']
        total_power_used = np.sum(phi)
        print(f"   ⚡ Potencia Total en el sistema: {total_power_used:.4f}")

if __name__ == "__main__":
    test_allocation_logic()