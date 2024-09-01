import itertools
import os
import time

seed_value = 1
p_list = [0.30, 0.50]
criteria = ["random", "last_layer"]



os.makedirs('./terminals', exist_ok=True)

name = "ResNet56_50_150ep"

for p, criterion in itertools.product(p_list, criteria):
    command = f"python Reinicializar_pesos_layers_ResNet56.py --seed_value {seed_value} --name {name} --p_reinitialized {p} --criterion_layer {criterion}  > ./terminals/{name}_{seed_value}_p_{p}_{criterion}.output"

    print(f"Executing: {command}")
    
    start_time = time.time()
    os.system(command)
    end_time = time.time()

    print(f"Time : {end_time - start_time:.2f} seconds\n")
    


name = "ResNet110_50_150ep"

for p, criterion in itertools.product(p_list, criteria):
    command = f"python Reinicializar_pesos_layers_ResNet110.py --seed_value {seed_value} --name {name} --p_reinitialized {p} --criterion_layer {criterion}  > ./terminals/{name}_{seed_value}_p_{p}_{criterion}.output"

    print(f"Executing: {command}")
    
    start_time = time.time()
    os.system(command)
    end_time = time.time()

    print(f"Time : {end_time - start_time:.2f} seconds\n")


seed_value = 2
p_list = [0.30, 0.50]
criteria = ["random", "last_layer"]



os.makedirs('./terminals', exist_ok=True)

name = "ResNet56_50_150ep"

for p, criterion in itertools.product(p_list, criteria):
    command = f"python Reinicializar_pesos_layers_ResNet56.py --seed_value {seed_value} --name {name} --p_reinitialized {p} --criterion_layer {criterion}  > ./terminals/{name}_{seed_value}_p_{p}_{criterion}.output"

    print(f"Executing: {command}")
    
    start_time = time.time()
    os.system(command)
    end_time = time.time()

    print(f"Time : {end_time - start_time:.2f} seconds\n")
    


name = "ResNet110_50_150ep"

for p, criterion in itertools.product(p_list, criteria):
    command = f"python Reinicializar_pesos_layers_ResNet110.py --seed_value {seed_value} --name {name} --p_reinitialized {p} --criterion_layer {criterion}  > ./terminals/{name}_{seed_value}_p_{p}_{criterion}.output"

    print(f"Executing: {command}")
    
    start_time = time.time()
    os.system(command)
    end_time = time.time()

    print(f"Time : {end_time - start_time:.2f} seconds\n")