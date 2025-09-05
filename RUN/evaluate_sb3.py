# ==================== evaluate_sb3.py ====================
from stable_baselines3 import PPO
from envs import WifiResourceEnv
import argparse, os

def evaluate(args):
    env = WifiResourceEnv(num_links=args.num_links, num_channels=args.num_channels,
                          p0=args.p0, sigma=args.sigma, lambda_power=args.lambda_power,
                          max_steps=args.max_steps, use_synthetic=bool(args.synthetic),
                          building_id=args.building_id, b5g=bool(args.b5g), seed=args.seed,
                          discrete_actions=bool(args.discrete_actions), num_power_levels=args.num_power_levels)
    model_path = args.model_path or os.path.join("runs", "ppo_wifi", "best_model")
    model = PPO.load(model_path, env=env, device="cpu")
    obs, info = env.reset(seed=args.seed); ep_reward = 0.0
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated: break
    print(f"Reward total: {ep_reward:.2f} | sum_rate: {info['sum_rate']:.2f} | exceso_pot: {info['power_excess']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_links", type=int, default=5)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--p0", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--lambda_power", type=float, default=1.0)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--building_id", type=int, default=990)
    parser.add_argument("--b5g", type=int, default=0)
    parser.add_argument("--discrete_actions", type=int, default=1)
    parser.add_argument("--num_power_levels", type=int, default=3)
    args = parser.parse_args(); evaluate(args)