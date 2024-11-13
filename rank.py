import torch
import torchvision
from torchvision import transforms
import numpy as np
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
import argparse
model_dict = {
        'resnet20': resnet20(),
        'resnet32': resnet32(),
        'resnet44': resnet44(),
        'resnet56': resnet56(),
        'resnet110': resnet110()
    }
parser = argparse.ArgumentParser(description='Analyze rank of features in ResNet layers')
parser.add_argument('--model_name', choices=model_dict, default='resnet18', help='ResNet model to analyze')
parser.add_argument('--model_path', type=str, default='path/to/your/model.pth', help='Path to the saved model file')
args = parser.parse_args()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model

    
model = model_dict[args.model_name].to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))  # Load weights from file
model = model.to(device)
model.eval()

# Load dataset with the same transforms as main.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Get one batch
images, _ = next(iter(trainloader))
images = images.to(device)

# Dictionary to store activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks for layers to analyze
hooks = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        hooks.append(module.register_forward_hook(get_activation(name)))

# Forward pass
with torch.no_grad():
    model(images)

# Calculate rank for each layer
ranks = {}
for name, activation in activations.items():
    # Reshape activation to 2D matrix (batch_size * all_other_dims, features)

    # Reshape activation to 2D matrix (batch, flattened_features)
    if len(activation.shape) > 2:  # Multi-dimensional layer (Conv, etc)
        features = activation.view(activation.shape[0], -1)
    else:  # 2D layer (Linear)
        features = activation
    # Covariance using torch
    cov_matrix = torch.cov(features.T)
    
    # SVD using torch 
    U, S, Vh = torch.linalg.svd(cov_matrix)
    
    # Calculate effective rank (number of singular values above threshold)
    threshold = 1e-3
    effective_rank = torch.sum(S > threshold)
    
    ranks[name] = {
        'total_dimensions': cov_matrix.shape[0],
        'effective_rank': effective_rank,
        'singular_values': S
    }

# Print results
for layer_name, rank_info in ranks.items():
    print(f"\nLayer: {layer_name}")
    print(f"Total dimensions: {rank_info['total_dimensions']}")
    print(f"Effective rank: {rank_info['effective_rank']}")
    print(f"Top 5 singular values: {rank_info['singular_values'][:5]}")

# Clean up hooks
for hook in hooks:
    hook.remove()
