import itertools
import os
import time


model_names = ["vgg11_bn", "vgg16_bn", "vgg19_bn"]
seeds = [1,2,3,4,5]
p_reinitializied_list = [0.5]
criterion = "last_layer"
os.makedirs('./terminals', exist_ok=True)

training_script = "main_vgg.py"

for model_name, seed, p_reinitializied in itertools.product(model_names, seeds, p_reinitializied_list):

    initial_lr = 0.01
    epoch = 200

    output_file = f"./terminals/{model_name}_seed{seed}_lr{initial_lr}_epochs{epoch}_p_{p_reinitializied_list}_{criterion}.txt"
    model_path = f"./pre-trained-models/{model_name}_seed_{seed}_lr_0.01_wd_0.0001_epochs_200.pth"
    command = (
        f"python {training_script} "
        f"--model_name {model_name} "
        f"--seed {seed} "
        f"--initial_lr {initial_lr} "
        f"--p_reinitialized {p_reinitializied} "
        f"--load_model {model_path} "
        f"> {output_file} 2>&1 "  # Redirect both stdout and stderr to the output file
        f"--reinitilization_epochs 0"
    )

    print(f"Executing: {command}", flush=True)

    start_time = time.time()

    os.system(command)
    
    end_time = time.time()
 

    print(f"Time: {end_time - start_time:.2f} seconds\n", flush=True)