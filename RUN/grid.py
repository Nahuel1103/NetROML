import os
import itertools

# Define los valores que quieres probar
eps_values = [ 5e-4, 1e-5, 5e-5]
mu_lr_values = [5e-4, 1e-5, 5e-5]

# Puedes ajustar los valores de arriba según lo que quieras explorar

total = 0
for i, (eps, mu_lr) in enumerate(itertools.product(eps_values, mu_lr_values)):
    print(f"Combinación {i+1}: eps={eps}, mu_lr={mu_lr}")
    cmd = (
        f"python /Users/nahuelpineyro/NetROML/RUN/train.py "
        f"--eps {eps} --mu_lr {mu_lr} --synthetic 0 --epochs 100 "
        f"--building_id 990 --b5g 0 --num_links 5 --num_layers 3 --k 3 --batch_size 64"
    )
    print(f"Ejecutando: {cmd}")
    os.system(cmd)
    total += 1

print(f"Total de combinaciones ejecutadas: {total}")