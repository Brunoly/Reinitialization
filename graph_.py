"""
This script creates and save graphs from the json of a all the models in the model folder.
"""


import os
import json
import matplotlib.pyplot as plt

def plot_metrics(history, save_path):
    plt.figure(figsize=(15, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot training and test accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_folders():
    root_folder = './folder'
    for model_folder in os.listdir(root_folder):
        model_path = os.path.join(root_folder, model_folder)
        if os.path.isdir(model_path):
            graphs_folder = os.path.join(model_path, 'graphs')
            if not os.path.exists(graphs_folder):
                os.makedirs(graphs_folder)
                
            for file in os.listdir(model_path):
                if file.endswith('_history.json'):
                    json_path = os.path.join(model_path, file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    history = data['history']
                    config = data['config']
                    
                    graph_name = f"{config['model_name']}_lr{config['initial_lr']}_epochs{config['n_epochs']}_graph.png"
                    graph_path = os.path.join(graphs_folder, graph_name)
                    
                    if not os.path.exists(graph_path):
                        plot_metrics(history, graph_path)
                        print(f"Created graph: {graph_path}")

if __name__ == "__main__":
    process_folders()