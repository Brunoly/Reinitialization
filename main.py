import argparse
import numpy as np
from Reinitialization.layer_masks import layer_to_reinitizalize
from Reinitialization.score_functions import random_score, last_layer
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

def parse_args():

    parser = argparse.ArgumenstParser(description='Train ResNet models on CIFAR-10')
    parser.add_argument('--initial_lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--model_name', type=str, default='resnet20', help='Model name (e.g., resnet20, resnet32)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--reinitilization_epochs', nargs='+', type=int, default=[999**999],
                        help='Epochs in which reinitilization happens')
    parser.add_argument('--p_reinitialized', type=float, default=0)
    parser.add_argument('--criterion_layer', type=str, default='random')

    # New arguments
    parser.add_argument('--save_epochs', nargs='+', type=int, default=[], help='Epochs at which to save the model')
    parser.add_argument('--load_model', type=str, default='', help='Path to a saved model to load')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch number to start training from')

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
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print(f"Loaded model from {args.load_model}")

    score_functions = {
        'last_layers': last_layer,
        'random': random_score
    }
    score_func = score_functions[args.criterion_layer]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=0.9, weight_decay=args.weight_decay)

    def lr_schedule(epoch):
        comeco = max([0] + [x for x in args.reinitilization_epochs if x < epoch])
        fim = min([x for x in args.reinitilization_epochs if x > epoch] + [args.n_epochs])

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
    parent_dir = './folder'
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_dir = os.path.join(parent_dir, f'{args.model_name}_lr{args.initial_lr}_epochs{args.n_epochs}_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    print(f"Models will be saved in: {run_dir}")

    print(f"\nTraining {args.model_name} with initial learning rate {args.initial_lr} and weight decay {args.weight_decay}\n")
    print("Model Summary:")
    summary(model, input_size=(3, 32, 32))

    for epoch in range(args.start_epoch, args.n_epochs):

        if epoch in args.reinitilization_epochs:
            with torch.no_grad():
                model = reinitialize_weights(
                    model,
                    layer_to_reinitizalize(
                        model,
                        p_reinitialized=float(args.p_reinitialized),
                        architecture=args.model_name,
                        score_function=score_func
                    )
                )

        start_time = time.time()

        train_loss, train_acc = train(epoch, model, device, trainloader, optimizer, criterion)
        epoch_time = time.time() - start_time

        start_time = time.time()
        test_loss, test_acc = test(model, device, testloader, criterion)
        test_time = time.time() - start_time

        scheduler.step()

        print(f"Epoch: {epoch + 1}/{args.n_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Epoch Time: {epoch_time:.2f}s | Test Time: {test_time:.2f}s")

        # Save model at specified epochs
        if epoch in args.save_epochs:
            save_filename = os.path.join(run_dir, f'{args.model_name}_epoch{epoch + 1}_checkpoint.pth')
            torch.save(model.state_dict(), save_filename)
            print(f'Model saved at epoch {epoch + 1} to {save_filename}')

    final_model_filename = os.path.join(run_dir, f'{args.model_name}_final_model.pth')
    print(f'Saved final model at: {final_model_filename}')
    torch.save(model.state_dict(), final_model_filename)

def reinitialize_weights(model, layers_to_reinit):
    # TODO: Implement shrink and perturb if needed
    for name, module in model.named_modules():
        # Check if this module is a convolutional layer we want to reinitialize
        if isinstance(module, nn.Conv2d) and name in layers_to_reinit:
            # Reinitialize weights
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Check if this module is a batch normalization layer we want to reinitialize
        elif isinstance(module, nn.BatchNorm2d) and any(str(idx) in name for idx in layers_to_reinit):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        # TODO: Implement reinitialization for other types if needed

    return model

if __name__ == '__main__':
    train_resnets()