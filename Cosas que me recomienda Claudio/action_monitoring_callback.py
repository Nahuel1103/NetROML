"""
Callbacks para monitorear y analizar las decisiones del agente
Registra: acciones, canales, potencias, mu, y más
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import json
import os


class DetailedMonitoringCallback(BaseCallback):
    """
    Callback comprehensivo que registra TODO lo importante:
    - Acciones tomadas por cada AP
    - Canales seleccionados
    - Niveles de potencia
    - Multiplicadores mu
    - Power violations
    - Rewards
    
    Permite hacer análisis detallado después del entrenamiento
    """
    
    def __init__(
        self, 
        n_APs=5,
        num_channels=3,
        n_power_levels=2,
        log_freq=100,  # Guardar cada N steps (no todos para ahorrar memoria)
        save_dir='./logs/monitoring',
        verbose=0
    ):
        super().__init__(verbose)
        
        self.n_APs = n_APs
        self.num_channels = num_channels
        self.n_power_levels = n_power_levels
        self.log_freq = log_freq
        self.save_dir = save_dir
        
        # Crear directorio si no existe
        os.makedirs(save_dir, exist_ok=True)
        
        # Contadores
        self.step_count = 0
        self.episode_count = 0
        
        # === INFORMACIÓN POR STEP (sampled cada log_freq) ===
        self.step_data = {
            'timesteps': [],
            'actions': [],           # [n_APs] acciones raw
            'channels': [],          # [n_APs] canal elegido por cada AP
            'power_levels': [],      # [n_APs] nivel de potencia
            'active_aps': [],        # [n_APs] bool: AP encendido?
            'mu': [],                # [n_APs] multiplicadores
            'avg_power': [],         # [n_APs] potencia promedio
            'rewards': [],           # scalar reward
        }
        
        # === ESTADÍSTICAS AGREGADAS POR EPISODIO ===
        self.episode_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'final_mu': [],
            'channel_usage': [],     # Histograma de uso de canales
            'power_usage': [],       # Histograma de niveles de potencia
        }
        
        # === CONTADORES ACUMULATIVOS ===
        self.action_counts = np.zeros(1 + num_channels * n_power_levels, dtype=int)
        self.channel_counts = np.zeros(num_channels, dtype=int)
        self.power_counts = np.zeros(n_power_levels, dtype=int)
        
    def _on_step(self) -> bool:
        """
        Ejecutado después de cada env.step()
        Registra información detallada
        """
        self.step_count += 1
        
        # Solo guardar cada log_freq steps (para no llenar memoria)
        if self.step_count % self.log_freq != 0:
            return True
        
        # =====================================================================
        # OBTENER INFORMACIÓN DEL ENVIRONMENT
        # =====================================================================
        
        # Detectar tipo de env (VecEnv vs normal)
        if hasattr(self.training_env, 'get_attr'):
            # VecEnv
            mu = self.training_env.get_attr('mu_power')[0]
            power_history = self.training_env.get_attr('power_history')[0]
            canales = self.training_env.get_attr('canales')[0]
            potencias = self.training_env.get_attr('potencias')[0]
        else:
            # Env normal
            mu = self.training_env.mu_power
            power_history = self.training_env.power_history
            canales = self.training_env.canales
            potencias = self.training_env.potencias
        
        # Convertir tensors a numpy
        if not isinstance(mu, np.ndarray):
            mu = mu.detach().cpu().numpy()
        
        # =====================================================================
        # OBTENER ÚLTIMA ACCIÓN
        # =====================================================================
        
        # Las acciones están en self.locals (variables locales del loop de SB3)
        # Puede variar según el algoritmo, pero generalmente:
        actions = None
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if not isinstance(actions, np.ndarray):
                actions = actions.detach().cpu().numpy()
        elif 'action' in self.locals:
            actions = self.locals['action']
            if not isinstance(actions, np.ndarray):
                actions = actions.detach().cpu().numpy()
        
        # =====================================================================
        # DECODIFICAR ACCIONES → CANALES Y POTENCIAS
        # =====================================================================
        
        if actions is not None:
            # Asegurarse que sea array 1D
            if actions.ndim > 1:
                actions = actions.flatten()[:self.n_APs]
            
            # Acciones raw
            self.step_data['actions'].append(actions.copy())
            
            # Contar acciones
            for a in actions:
                if 0 <= a < len(self.action_counts):
                    self.action_counts[int(a)] += 1
            
            # Decodificar: canal y potencia
            channels_decoded = np.zeros(self.n_APs, dtype=int)
            powers_decoded = np.zeros(self.n_APs, dtype=int)
            active_mask = actions > 0
            
            if np.any(active_mask):
                actions_active = actions[active_mask] - 1
                channels_decoded[active_mask] = (actions_active // self.n_power_levels) + 1
                powers_decoded[active_mask] = actions_active % self.n_power_levels
                
                # Contar uso de canales y potencias
                for ch in channels_decoded[active_mask]:
                    self.channel_counts[int(ch) - 1] += 1
                for pw in powers_decoded[active_mask]:
                    self.power_counts[int(pw)] += 1
            
            self.step_data['channels'].append(channels_decoded.copy())
            self.step_data['power_levels'].append(powers_decoded.copy())
            self.step_data['active_aps'].append(active_mask.copy())
        
        # =====================================================================
        # GUARDAR OTRAS MÉTRICAS
        # =====================================================================
        
        self.step_data['timesteps'].append(self.step_count)
        self.step_data['mu'].append(mu.copy())
        
        # Potencia promedio
        if len(power_history) > 0:
            avg_power = np.mean(power_history, axis=0)
            self.step_data['avg_power'].append(avg_power.copy())
        
        # Reward
        if 'rewards' in self.locals:
            reward = self.locals['rewards']
            if not isinstance(reward, (int, float)):
                reward = float(reward.item()) if hasattr(reward, 'item') else float(reward)
            self.step_data['rewards'].append(reward)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Al final de cada rollout, imprimir resumen
        """
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"📊 RESUMEN - Step {self.step_count}")
            print(f"{'='*60}")
            
            # Acciones más comunes
            if len(self.action_counts) > 0 and np.sum(self.action_counts) > 0:
                action_probs = self.action_counts / np.sum(self.action_counts)
                top_actions = np.argsort(action_probs)[-3:][::-1]
                
                print("\n🎯 Top 3 acciones:")
                for i, action_idx in enumerate(top_actions):
                    print(f"  {i+1}. Acción {action_idx}: {action_probs[action_idx]*100:.1f}%")
            
            # Uso de canales
            if np.sum(self.channel_counts) > 0:
                channel_probs = self.channel_counts / np.sum(self.channel_counts)
                print("\n📡 Uso de canales:")
                for ch in range(self.num_channels):
                    print(f"  Canal {ch+1}: {channel_probs[ch]*100:.1f}%")
            
            # Mu actual
            if len(self.step_data['mu']) > 0:
                last_mu = self.step_data['mu'][-1]
                print(f"\n⚡ Mu actual: {last_mu}")
            
            print(f"{'='*60}\n")
    
    def save_data(self):
        """
        Guarda todos los datos recolectados en archivos
        """
        # Guardar datos de steps como numpy
        np.savez(
            os.path.join(self.save_dir, 'step_data.npz'),
            **self.step_data
        )
        
        # Guardar contadores
        np.savez(
            os.path.join(self.save_dir, 'counters.npz'),
            action_counts=self.action_counts,
            channel_counts=self.channel_counts,
            power_counts=self.power_counts
        )
        
        # Guardar metadata
        metadata = {
            'n_APs': self.n_APs,
            'num_channels': self.num_channels,
            'n_power_levels': self.n_power_levels,
            'total_steps': self.step_count,
            'log_freq': self.log_freq
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Datos guardados en {self.save_dir}")
    
    def _on_training_end(self) -> None:
        """
        Al final del entrenamiento, guardar todo
        """
        self.save_data()
        self.plot_analysis()
    
    def plot_analysis(self):
        """
        Genera gráficas de análisis
        """
        if len(self.step_data['timesteps']) == 0:
            print("⚠️ No hay datos para graficar")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Entrenamiento', fontsize=16, fontweight='bold')
        
        timesteps = self.step_data['timesteps']
        
        # =====================================================================
        # 1. EVOLUCIÓN DE MU
        # =====================================================================
        if len(self.step_data['mu']) > 0:
            mu_array = np.array(self.step_data['mu'])
            for i in range(min(self.n_APs, mu_array.shape[1])):
                axes[0, 0].plot(timesteps, mu_array[:, i], label=f'AP {i}', alpha=0.7)
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Mu')
            axes[0, 0].set_title('Evolución de Multiplicadores μ')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # 2. DISTRIBUCIÓN DE ACCIONES
        # =====================================================================
        if np.sum(self.action_counts) > 0:
            action_probs = self.action_counts / np.sum(self.action_counts)
            axes[0, 1].bar(range(len(action_probs)), action_probs)
            axes[0, 1].set_xlabel('Acción')
            axes[0, 1].set_ylabel('Frecuencia')
            axes[0, 1].set_title('Distribución de Acciones')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # 3. USO DE CANALES
        # =====================================================================
        if np.sum(self.channel_counts) > 0:
            channel_probs = self.channel_counts / np.sum(self.channel_counts)
            colors = plt.cm.viridis(np.linspace(0, 1, self.num_channels))
            axes[1, 0].bar(range(1, self.num_channels + 1), channel_probs, color=colors)
            axes[1, 0].set_xlabel('Canal')
            axes[1, 0].set_ylabel('Frecuencia de uso')
            axes[1, 0].set_title('Distribución de Canales')
            axes[1, 0].set_xticks(range(1, self.num_channels + 1))
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # 4. USO DE NIVELES DE POTENCIA
        # =====================================================================
        if np.sum(self.power_counts) > 0:
            power_probs = self.power_counts / np.sum(self.power_counts)
            axes[1, 1].bar(range(self.n_power_levels), power_probs, color='coral')
            axes[1, 1].set_xlabel('Nivel de Potencia')
            axes[1, 1].set_ylabel('Frecuencia de uso')
            axes[1, 1].set_title('Distribución de Niveles de Potencia')
            axes[1, 1].set_xticks(range(self.n_power_levels))
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # 5. APs ACTIVOS vs APAGADOS
        # =====================================================================
        if len(self.step_data['active_aps']) > 0:
            active_array = np.array(self.step_data['active_aps'])
            avg_active = np.mean(active_array, axis=1)  # Promedio por step
            axes[2, 0].plot(timesteps, avg_active, color='green', linewidth=2)
            axes[2, 0].set_xlabel('Steps')
            axes[2, 0].set_ylabel('Fracción de APs activos')
            axes[2, 0].set_title('APs Activos a lo largo del tiempo')
            axes[2, 0].set_ylim([0, 1.1])
            axes[2, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # 6. REWARDS
        # =====================================================================
        if len(self.step_data['rewards']) > 0:
            rewards = self.step_data['rewards']
            # Suavizar con moving average
            window = min(20, len(rewards) // 10)
            if window > 1:
                rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                timesteps_smooth = timesteps[:len(rewards_smooth)]
                axes[2, 1].plot(timesteps_smooth, rewards_smooth, color='blue', linewidth=2)
            else:
                axes[2, 1].plot(timesteps, rewards, color='blue', alpha=0.5)
            axes[2, 1].set_xlabel('Steps')
            axes[2, 1].set_ylabel('Reward')
            axes[2, 1].set_title('Evolución de Rewards (smoothed)')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_analysis.png'), dpi=150)
        print(f"📈 Gráficas guardadas en {self.save_dir}/training_analysis.png")
        plt.close()


class SimplifiedMonitoringCallback(BaseCallback):
    """
    Versión SIMPLIFICADA si solo quieres lo esencial
    """
    def __init__(self, n_APs=5, num_channels=3, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.n_APs = n_APs
        self.num_channels = num_channels
        self.log_freq = log_freq
        self.step_count = 0
        
        # Solo lo básico
        self.mu_history = []
        self.channel_usage = np.zeros(num_channels, dtype=int)
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.log_freq == 0:
            # Obtener mu
            if hasattr(self.training_env, 'get_attr'):
                mu = self.training_env.get_attr('mu_power')[0]
                canales = self.training_env.get_attr('canales')[0]
            else:
                mu = self.training_env.mu_power
                canales = self.training_env.canales
            
            if not isinstance(mu, np.ndarray):
                mu = mu.detach().cpu().numpy()
            
            self.mu_history.append(mu.copy())
            
            # Contar canales activos
            for ch in canales:
                if ch > 0:
                    self.channel_usage[int(ch) - 1] += 1
            
            if self.verbose > 0:
                print(f"Step {self.step_count} | Mu: {mu} | Canales: {canales}")
        
        return True


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def ejemplo_uso():
    """
    Cómo usar estos callbacks en tu entrenamiento
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    
    # Crear environment (simulado)
    # env = APNetworkEnv(...)
    
    # =========================================================================
    # OPCIÓN 1: Monitoring comprehensivo
    # =========================================================================
    detailed_callback = DetailedMonitoringCallback(
        n_APs=5,
        num_channels=3,
        n_power_levels=2,
        log_freq=100,  # Guardar cada 100 steps
        save_dir='./logs/detailed',
        verbose=1
    )
    
    # =========================================================================
    # OPCIÓN 2: Monitoring simplificado
    # =========================================================================
    simple_callback = SimplifiedMonitoringCallback(
        n_APs=5,
        num_channels=3,
        log_freq=1000,
        verbose=1
    )
    
    # =========================================================================
    # COMBINAR CON OTROS CALLBACKS
    # =========================================================================
    callbacks = [
        detailed_callback,  # O simple_callback
        EvalCallback(...),
        CheckpointCallback(...)
    ]
    
    # Entrenar
    model = PPO(...)
    model.learn(total_timesteps=100000, callback=callbacks)
    
    # =========================================================================
    # ANÁLISIS POST-ENTRENAMIENTO
    # =========================================================================
    
    # Si usaste DetailedMonitoringCallback, ya tienes las gráficas
    # También puedes hacer análisis custom:
    
    # Cargar datos guardados
    data = np.load('./logs/detailed/step_data.npz')
    counters = np.load('./logs/detailed/counters.npz')
    
    # Análisis personalizado
    actions = data['actions']
    channels = data['channels']
    mu = data['mu']
    
    # Ejemplo: Ver qué canal prefiere cada AP
    for ap in range(5):
        ap_channels = channels[:, ap]
        unique, counts = np.unique(ap_channels[ap_channels > 0], return_counts=True)
        print(f"AP {ap} - Canal favorito: {unique[np.argmax(counts)]}")


if __name__ == "__main__":
    print(__doc__)
    print("\nPara ejecutar ejemplo, llama a ejemplo_uso()")
