###TODO verificar que isso esta certo e que é o jeito certo de fazer linear probing (só faz uma época/batch pelo que vi)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
import argparse

import torch
import torchvision.transforms as transforms
import numpy as np

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

#transform_train = transform_test


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model_dict = {
        'resnet20': resnet20(),
        'resnet32': resnet32(),
        'resnet44': resnet44(),
        'resnet56': resnet56(),
        'resnet110': resnet110()
    }

class LinearProbe(nn.Module):
    def __init__(self, in_features):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(in_features, 10)
        
        # Good initialization is still useful
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

def get_layer_activations(model, layer_name):
    # Use a dictionary instead of a list to store activations
    activations_dict = {}
    
    def hook(module, input, output):
        # Store each batch's activation separately
        activations_dict['current'] = output.detach()
    
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            return handle, activations_dict
    
    return None, None

def train_probe(probe, activations, labels, device, batch_size=64, epochs=15):        
    probe = probe.to(device)
    
    if len(activations.shape) > 2:
        activations = activations.view(activations.size(0), -1)
    
    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(activations, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Debug prints
    print(f"Training data shape: {activations.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels in training: {torch.unique(labels).tolist()}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    def check_predictions(all_preds):
        unique_preds, counts = torch.unique(all_preds, return_counts=True)
        print(f"Unique predictions: {unique_preds.tolist()}")
        print(f"Prediction counts: {counts.tolist()}")
    
    for epoch in range(epochs):
        probe.train()
        total_correct = 0
        total_samples = 0
        all_preds = []
        
        for batch_activations, batch_labels in dataloader:
            batch_activations = batch_activations.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_activations)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            _, preds = outputs.max(1)
            total_correct += preds.eq(batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            all_preds.append(preds.cpu())
            
        epoch_accuracy = 100. * total_correct / total_samples
        
        if epoch == 0 or epoch == epochs-1:
            print(f"\nEpoch {epoch}")
            check_predictions(torch.cat(all_preds))
            
    return epoch_accuracy

def evaluate_probe(probe, activations, labels, device, batch_size=512):
    probe = probe.to(device)
    
    if len(activations.shape) > 2:
        activations = activations.view(activations.size(0), -1)
    
    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(activations, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nEvaluation:")
    print(f"Test data shape: {activations.shape}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Unique labels in test: {torch.unique(labels).tolist()}")
    
    probe.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    
    with torch.no_grad():
        for batch_activations, batch_labels in dataloader:
            batch_activations = batch_activations.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = probe(batch_activations)
            _, preds = outputs.max(1)
            
            total_correct += preds.eq(batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            all_preds.append(preds.cpu())
    
    all_preds = torch.cat(all_preds)
    unique_preds, counts = torch.unique(all_preds, return_counts=True)
    print(f"Unique test predictions: {unique_preds.tolist()}")
    print(f"Test prediction counts: {counts.tolist()}")
    
    accuracy = 100. * total_correct / total_samples
    return accuracy


def probe_model(model, batch_size=512, seed=42, save_probes=False, probes_dir='saved_probes'):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    layer_names = [name for name, _ in model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))]
    
    result = {}
    for layer_name in layer_names:
        handle, activations_dict = get_layer_activations(model, layer_name)
        if handle is None:
            continue
            
        # Get one batch to print layer shape
        images, labels = next(iter(trainloader))
        images = images.to(device)
        _ = model(images)
        layer_output = activations_dict['current']
        print(f"\nLayer: {layer_name}")
        print(f"Layer output shape: {layer_output.shape}")
        
        train_activations = []
        train_labels_list = []
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            _ = model(images)
            # Access the current batch's activation from the dictionary
            train_activations.append(activations_dict['current'])
            train_labels_list.append(labels)
        
        train_activations = torch.cat(train_activations, dim=0)
        train_labels = torch.cat(train_labels_list, dim=0)
        
        test_activations = []
        test_labels_list = []
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            _ = model(images)
            test_activations.append(activations_dict['current'])
            test_labels_list.append(labels)
        
        test_activations = torch.cat(test_activations, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)
        
        in_features = train_activations.view(train_activations.size(0), -1).size(1)
        probe = LinearProbe(in_features)
        print(f"Probe input features: {in_features}")
        print(f"Probe architecture:")
        print(probe)
        
        train_accuracy = train_probe(probe, train_activations, train_labels, device)
        test_accuracy = evaluate_probe(probe, test_activations, test_labels, device)
        
        if save_probes:
            os.makedirs(probes_dir, exist_ok=True)
            probe_path = os.path.join(probes_dir, f'probe_{layer_name.replace(".", "_")}.pth')
            torch.save(probe.state_dict(), probe_path)
            print(f"Saved probe for layer {layer_name} to {probe_path}")
        

        print(f"Train accuracy: {train_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print("-" * 50)
        
        result[layer_name] = test_accuracy
        handle.remove()
        torch.cuda.empty_cache()
    return result

def print_probe_matrix(probe):
    """Print the first 10 column vectors of the probe's linear layer weights"""
    with torch.no_grad():
        weights = probe.linear.weight.data
        num_cols = min(10, weights.size(1))
        print("\nProbe Weight Matrix (first 10 columns):")
        print("-" * 50)
        for row in weights[:, :num_cols]:
            # Format each weight in scientific notation with 3 decimal places
            formatted_row = [f"{x:.3e}" for x in row.tolist()]
            print(formatted_row)
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet20')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    
    model = model_dict[args.model_name].to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_path))
    
    result = probe_model(model, args.batch_size, args.seed)
    print(f"\nProbing {args.model_name} layers:")
    print("-" * 50)
    for layer_name, accuracy in result.items():
        print(f"Layer: {layer_name:<30} Test Accuracy: {accuracy:.2f}%")
        
        handle, activations_dict = get_layer_activations(model, layer_name)
        if handle:
            # Get one batch of data
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            images, labels = next(iter(trainloader))
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            _ = model(images)
            
            # Access the current batch's activation from the dictionary
            current_activation = activations_dict['current']
            in_features = current_activation.view(current_activation.size(0), -1).size(1)
            probe = LinearProbe(in_features)
            train_probe(probe, current_activation, labels, 'cuda' if torch.cuda.is_available() else 'cpu')
            print_probe_matrix(probe)
            
            handle.remove()
    
    return result

if __name__ == '__main__':
    main()
