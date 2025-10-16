H_list = generar_matrices_H(num_episodios=1000)
H_iter = iter(H_list)

obs, _ = env.reset()
for step in range(max_steps):
    action = policy(obs)
    H = next(H_iter)  # ⬅️ sacás la siguiente matriz
    obs, reward, done, trunc, info = env.step(action, H=H)
