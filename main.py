import argparse
import numpy as np
from layer_masks import layer_to_reinitizalize
from score_functions import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
import time
import os
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from probe import probe_model
import json
import re

def parse_args():

    parser = argparse.ArgumentParser(description='Train ResNet models on CIFAR-10')
    parser.add_argument('--initial_lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--model_name', type=str, default='resnet20', help='Model name (e.g., resnet20, resnet32)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--reinitilization_epochs', nargs='+', type=int, default=[-1],
                        help='Epochs in which reinitilization happens')
    parser.add_argument('--num_layers_reinitialize', type=int, default=0)
    parser.add_argument('--criterion_layer', type=str, default='random')

    # New arguments
    parser.add_argument('--save_epochs', nargs='+', type=int, default=[], help='Epochs at which to save the model')
    parser.add_argument('--load_model', type=str, default='', help='Path to a saved model to load')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch number to start training from')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--save_probes', action='store_true', help='Whether to save the probes')
    parser.add_argument('--probes_dir', type=str, default='saved_probes', help='Directory to save probes')
    parser.add_argument('--keep_ratio', type=float, default=0.0, help='Ratio of weights to keep during reinitialization (0.0 to 1.0)') 
    return parser.parse_args()
    

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])


# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch, model, device, trainloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(trainloader)
    return avg_train_loss, train_acc

def test(model, device, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    avg_test_loss = test_loss / len(testloader)
    return avg_test_loss, test_acc

def train_resnets():
    args = parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_dict = {
        'resnet20': resnet20(),
        'resnet32': resnet32(),
        'resnet44': resnet44(),
        'resnet56': resnet56(),
        'resnet110': resnet110()
    }
    
    model = model_dict[args.model_name].to(device)

    # Load external model if provided
    loaded_epochs = 0
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print(f"Loaded model from {args.load_model}")
        match = re.search(r'epochs(\d+)', args.load_model) #????
        if match:
            loaded_epochs = int(match.group(1))

        initial_test_loss, initial_test_acc = test(model, device, testloader, criterion)
        print(f"\nInitial Model Performance:")
        print(f"Test Accuracy: {initial_test_acc:.2f}%")
        print(f"Test Loss: {initial_test_loss:.4f}\n")

    score_functions = {
        'last_layers': last_layer,
        'random': random_score,
        "tunnel": last_layer,
        "one_layer": one_layer
    }
    score_func = score_functions[args.criterion_layer]

    if args.criterion_layer == 'tunnel':
        probing_results = probe_model(model, save_probes=args.save_probes, probes_dir=args.probes_dir)        threshold = 0.95 * max(probing_results.values()) #ou probing_results.values()[-1] ???
        layers_ordered = sorted(probing_results.keys(), key=lambda x: probing_results[x], reverse=True)
        first_above = [i for i, layer in enumerate(layers_ordered) if probing_results[layer] >= threshold][0]
        args.num_layers_reinitialize = len(layers_ordered) - first_above

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=0.9, weight_decay=args.weight_decay)

    def lr_schedule(epoch):
        comeco = max([0] + [x for x in args.reinitilization_epochs if x < epoch])
        fim = min([x for x in args.reinitilization_epochs if x > epoch] + [args.n_epochs])
        #o janela começa logo depois do reinitialization epoch

        epoch_relativo = epoch - comeco
        tamanho_janela_de_treino = fim - comeco

        if epoch_relativo < tamanho_janela_de_treino * 0.5:
            return 1
        elif epoch_relativo < tamanho_janela_de_treino * 0.75:
            return 0.1
        else:
            return 0.01

    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule, last_epoch=args.start_epoch - 1)

    # Create folder structure to save models
    parent_dir = './modelos'
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_dir = os.path.join(parent_dir, f'{args.model_name}_lr{args.initial_lr}_epochs{args.n_epochs}_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    print(f"Models will be saved in: {run_dir}")

    print(f"\nTraining {args.model_name} with initial learning rate {args.initial_lr} and weight decay {args.weight_decay}\n")
    print("Model Summary:")
    summary(model, input_size=(3, 32, 32))

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'lr': []
    }

    for epoch in range(args.start_epoch, args.n_epochs):
        if epoch in args.reinitilization_epochs:
            with torch.no_grad():
                model = reinitialize_weights(
                    model,
                    layer_to_reinitizalize(
                        model,
                        num_layers_reinitialize=int(args.num_layers_reinitialize),
                        architecture=args.model_name,
                        score_function=score_func
                    ),
                    keep_ratio=args.keep_ratio
                )


        start_time = time.time()

        train_loss, train_acc = train(epoch, model, device, trainloader, optimizer, criterion)
        epoch_time = time.time() - start_time

        start_time = time.time()
        test_loss, test_acc = test(model, device, testloader, criterion)
        test_time = time.time() - start_time

        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        print(f"Epoch: {epoch + 1}/{args.n_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Learning Rate: {current_lr:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s | Test Time: {test_time:.2f}s")

        # Save model at specified epochs
        if epoch in args.save_epochs:
            save_filename = os.path.join(run_dir, f'{args.model_name}_seed{args.seed}_lr{args.initial_lr}_epochs+{loaded_epochs}+{args.n_epochs}_{timestamp}_criteria{args.criterion_layer}_epoch{epoch + 1}_checkpoint.pth')
            torch.save(model.state_dict(), save_filename)
            print(f'Model saved at epoch {epoch + 1} to {save_filename}')

    final_model_filename = os.path.join(run_dir, f'{args.model_name}_seed{args.seed}_lr{args.initial_lr}_epochs{loaded_epochs}+{args.n_epochs}_criteria{args.criterion_layer}_final_model.pth')
    print(f'Saved final model at: {final_model_filename}')
    torch.save(model.state_dict(), final_model_filename)

    # Save the history and config to JSON
    config = vars(args)
    history_data = {
        "config": config,
        "history": history
    }

    json_filename = f"{args.model_name}_seed{args.seed}_lr{args.initial_lr}_epochs{loaded_epochs}+{args.n_epochs}_criteria{args.criterion_layer}_history.json"
    json_path = os.path.join(run_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(history_data, f, indent=4)
    print(f"Training history saved to {json_path}") 

def reinitialize_weights(model, layers_to_reinit, keep_ratio=0):
    print("\nReinitializing Weights:")
    print(f"Layers to reinitialize: {layers_to_reinit}")
    print(f"Keep ratio: {keep_ratio}")
    
    # Track statistics for debugging
    stats = {
        'conv_layers_modified': 0,
        'bn_layers_modified': 0,
        'weight_stats': {},
        'bias_stats': {}
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in layers_to_reinit:
            stats['conv_layers_modified'] += 1
            print(f"\nReinitializing Conv2d Layer: {name}")
            print(f"Shape: {module.weight.shape}")
            
            # Weight reinitialization
            original_weights = module.weight.data.clone()
            new_weights = torch.empty_like(module.weight)
            nn.init.kaiming_normal_(new_weights, mode='fan_out', nonlinearity='relu')
            
            # Calculate and store statistics
            weight_stats = {
                'original_mean': original_weights.mean().item(),
                'original_std': original_weights.std().item(),
                'new_mean': new_weights.mean().item(),
                'new_std': new_weights.std().item()
            }
            
            # Mix weights according to keep_ratio
            module.weight.data = keep_ratio * original_weights + (1 - keep_ratio) * new_weights
            
            weight_stats['final_mean'] = module.weight.data.mean().item()
            weight_stats['final_std'] = module.weight.data.std().item()
            
            stats['weight_stats'][name] = weight_stats
            
            print("Weight Statistics:")
            print(f"Original - Mean: {weight_stats['original_mean']:.4f}, Std: {weight_stats['original_std']:.4f}")
            print(f"New      - Mean: {weight_stats['new_mean']:.4f}, Std: {weight_stats['new_std']:.4f}")
            print(f"Final    - Mean: {weight_stats['final_mean']:.4f}, Std: {weight_stats['final_std']:.4f}")
            
            # Bias reinitialization if present
            if module.bias is not None:
                original_bias = module.bias.data.clone()
                new_bias = torch.zeros_like(module.bias)
                
                bias_stats = {
                    'original_mean': original_bias.mean().item(),
                    'original_std': original_bias.std().item(),
                    'new_mean': new_bias.mean().item(),
                    'new_std': new_bias.std().item()
                }
                
                module.bias.data = keep_ratio * original_bias + (1 - keep_ratio) * new_bias
                
                bias_stats['final_mean'] = module.bias.data.mean().item()
                bias_stats['final_std'] = module.bias.data.std().item()
                
                stats['bias_stats'][name] = bias_stats
                
                print("\nBias Statistics:")
                print(f"Original - Mean: {bias_stats['original_mean']:.4f}, Std: {bias_stats['original_std']:.4f}")
                print(f"New      - Mean: {bias_stats['new_mean']:.4f}, Std: {bias_stats['new_std']:.4f}")
                print(f"Final    - Mean: {bias_stats['final_mean']:.4f}, Std: {bias_stats['final_std']:.4f}")

        elif isinstance(module, nn.BatchNorm2d) and any(str(idx) in name for idx in layers_to_reinit):
            stats['bn_layers_modified'] += 1
            print(f"\nReinitializing BatchNorm Layer: {name}")
            
            # Weight reinitialization
            original_weight = module.weight.data.clone()
            new_weight = torch.ones_like(module.weight)
            module.weight.data = keep_ratio * original_weight + (1 - keep_ratio) * new_weight
            
            # Bias reinitialization
            original_bias = module.bias.data.clone()
            new_bias = torch.zeros_like(module.bias)
            module.bias.data = keep_ratio * original_bias + (1 - keep_ratio) * new_bias
            
            print(f"Weight - Original mean: {original_weight.mean():.4f}, Final mean: {module.weight.data.mean():.4f}")
            print(f"Bias   - Original mean: {original_bias.mean():.4f}, Final mean: {module.bias.data.mean():.4f}")
    
    # Print summary statistics
    print("\nReinitialization Summary:")
    print(f"Conv layers modified: {stats['conv_layers_modified']}")
    print(f"BatchNorm layers modified: {stats['bn_layers_modified']}")
    
    if stats['conv_layers_modified'] == 0:
        print("WARNING: No Conv2d layers were reinitialized!")
    
    return model

if __name__ == '__main__':
    train_resnets()