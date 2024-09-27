import itertools
import os
import time


model_names = ["vgg11", "vgg16", "vgg19"]
seeds = [1]

os.makedirs('./terminals', exist_ok=True)

training_script = "main.py"

for model_name, seed in itertools.product(model_names, seeds):

    initial_lr = 0.1
    epoch = 200

    output_file = f"./terminals/{model_name}_seed{seed}_lr{initial_lr}_epochs{epoch}.txt"

    command = (
        f"python {training_script} "
        f"--model_name {model_name} "
        f"--seed {seed} "
        f"--initial_lr {initial_lr} "
        f"> {output_file} 2>&1"  # Redirect both stdout and stderr to the output file
    )

    print(f"Executing: {command}", flush=True)

    start_time = time.time()

    os.system(command)
    
    end_time = time.time()
 

    print(f"Time: {end_time - start_time:.2f} seconds\n", flush=True)