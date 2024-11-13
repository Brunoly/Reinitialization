
import torch
import matplotlib.pyplot as plt
from resnet import resnet110

def generate_noise_image(size=32):
    noise = torch.randn(3, 3, size, size)  
    return noise
def main():
    # Create model
    model = resnet110()
    
    # Load model weights from checkpoint
    checkpoint_path = "/home/vm02/Desktop/raul/Reinitialization/resnet110_seed_1_lr_0.1_wd_0.0_epochs_200.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Generate random noise
    noise_image = generate_noise_image()
    
    # Forward pass
    with torch.no_grad():
        output = model(noise_image)
        
    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Convert to numpy for display
    probabilities = probabilities.numpy()[0]
    
    # Display results
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 5))
    plt.bar(classes, probabilities)
    plt.xticks(rotation=45)
    plt.title('ResNet110 Predictions on Random Noise')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()
    
    # Print numerical values
    print("\nClass Probabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"{cls}: {prob:.4f}")

if __name__ == "__main__":
    main()
