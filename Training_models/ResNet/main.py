# Requires files resnet.pt and torchsummary.py

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary

from resnet import resnet20, resnet32, resnet44, resnet56, resnet110


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10')
    parser.add_argument('--initial_lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--model_name', type=str, default='resnet20', help='Model name (e.g., resnet20, resnet32)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='')

    return parser.parse_args()


# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    return test_acc


def train_resnets():
    args = parse_args()


    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model_dict = {
        'resnet20': resnet20(),
        'resnet32': resnet32(),
        'resnet44': resnet44(),
        'resnet56': resnet56(),
        'resnet110': resnet110()
    }
    model = model_dict[args.model_name].to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=0.9, weight_decay=args.weight_decay)

    def lr_schedule(epoch):
        if epoch < args.n_epochs * 0.5:
            return 1
        elif epoch < args.n_epochs * 0.75:
            return 0.1
        else:
            return 0.01

    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    print(f"\nTraining {args.model_name} with initial learning rate {args.initial_lr}, "
          f"weight decay {args.weight_decay} and seed {args.seed}\n")
    print("Model Summary:")
    summary(model, input_size=(3, 32, 32))

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'lr': []
    }

    for epoch in range(args.n_epochs):
        start_time = time.time()

        train_loss, train_acc = train(epoch, model, device, trainloader, optimizer, criterion)
        epoch_time = time.time() - start_time

        test_acc = test(model, device, testloader, criterion)

        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        print(f"Epoch: {epoch + 1}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Learning Rate: {current_lr:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s")

    # Save the model
    #current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('models', exist_ok=True)
    model_filename = f"{args.model_name}_seed_{args.seed}_lr_{args.initial_lr}_wd_{args.weight_decay}_epochs_{args.n_epochs}.pth"
    model_path = os.path.join('models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    config = vars(args)
    history_data = {
        "config": config,
        "history": history
    }

    # Save the history and config to JSON
    json_filename = f"{args.model_name}_seed_{args.seed}_lr_{args.initial_lr}_wd_{args.weight_decay}epochs_{args.n_epochs}_history.json"
    os.makedirs('history', exist_ok=True)
    json_path = os.path.join('history', json_filename)
    with open(json_path, 'w') as f:
        json.dump(history_data, f, indent=4)
    print(f"Training history saved to {json_path}")


if __name__ == '__main__':
    train_resnets()
