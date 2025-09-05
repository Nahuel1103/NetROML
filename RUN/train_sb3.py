# ==================== train_sb3.py ====================
import os
import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from envs import WifiResourceEnv
from policies import GNNExtractor 

def train(args):
    env = WifiResourceEnv(num_links=args.num_links, num_channels=args.num_channels,
                          p0=args.p0, sigma=args.sigma, lambda_power=args.lambda_power,
                          max_steps=args.max_steps, use_synthetic=bool(args.synthetic),
                          building_id=args.building_id, b5g=bool(args.b5g), seed=args.seed,
                          discrete_actions=bool(args.discrete_actions), num_power_levels=args.num_power_levels)

    run_dir = os.path.join("runs", "ppo_wifi")
    os.makedirs(run_dir, exist_ok=True)

    eval_env = WifiResourceEnv(num_links=args.num_links, num_channels=args.num_channels,
                               p0=args.p0, sigma=args.sigma, lambda_power=args.lambda_power,
                               max_steps=args.max_steps, use_synthetic=bool(args.synthetic),
                               building_id=args.building_id, b5g=bool(args.b5g), seed=args.seed,
                               discrete_actions=bool(args.discrete_actions), num_power_levels=args.num_power_levels)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(1000, args.total_timesteps // 20),
        deterministic=True
    )

    # 👇 definimos kwargs para cambiar MLP -> GNN
    if args.use_gnn:
        policy_kwargs = dict(
            features_extractor_class=GNNExtractor,
            features_extractor_kwargs=dict(hidden_dim=args.hidden_dim, num_layers=args.num_layers),
        )
    else:
        policy_kwargs = {}

    # device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=run_dir,
        policy_kwargs=policy_kwargs,
        device='cpu'
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    save_path = os.path.join(run_dir, "final_model")
    model.save(save_path)
    print(f"Modelo guardado en: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=2000)
    parser.add_argument("--num_links", type=int, default=5)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--p0", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--lambda_power", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--building_id", type=int, default=990)
    parser.add_argument("--b5g", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)

    # 👇 nuevos argumentos para el extractor GNN
    parser.add_argument("--use_gnn", type=int, default=1)  # 1=usar GNNExtractor, 0=usar MLP
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--discrete_actions", type=int, default=1)
    parser.add_argument("--num_power_levels", type=int, default=3)

    args = parser.parse_args()
    train(args)

