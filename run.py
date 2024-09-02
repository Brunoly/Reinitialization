import itertools
import os

seed_values = [1, 2]
p_list = [0.30, 0.50]
criteria = ["random", "last_layer"]
names = ["ResNet56_50_150ep", "ResNet110_50_150ep"]

os.makedirs('./terminals', exist_ok=True)

all_commands = []

for seed_value, name in itertools.product(seed_values, names):
    for p, criterion in itertools.product(p_list, criteria):
        resnet_type = "56" if "56" in name else "110"
        command = f"python Reinicializar_pesos_layers_ResNet{resnet_type}.py --seed_value {seed_value} --name {name} --p_reinitialized {p} --criterion_layer {criterion} --timeit > ./terminals/{name}_{seed_value}_p_{p}_{criterion}.output"
        all_commands.append(command)

# Join all commands with ' && ' to run them sequentially
full_command = " && ".join(all_commands)

print("Executing all commands:")
print(full_command)

os.system(full_command)