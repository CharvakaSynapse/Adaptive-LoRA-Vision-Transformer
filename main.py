import torch
import time
import numpy as np
import json
from data_utils import *
from model_utils import *
from train_utils import *
from eval_utils import *

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    start = time.time()
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(device, config)

    # Adaptive LoRA ranks
    adaptive_ranks = compute_adaptive_ranks(model, train_loader, device, config)

    # Apply LoRA
    model = apply_lora(model, adaptive_ranks, device, config)

    # Set trainable parameters
    set_trainable_parameters(model)

    # Train
    best_model_path = config['paths']['best_model']
    train_losses, val_losses, train_accuracies, val_accuracies, best_epoch = train_model(
        model, train_loader, val_loader, test_loader, device, best_model_path, config
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, val_loader, device, best_model_path, adaptive_ranks)
    
    # Plot
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, config)

    # Print final results
    print(f"Best Model (Epoch {best_epoch}): Test Loss: {metrics['test_loss']:.4f}, "
          f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"ECE: {metrics['ece']:.4f}, Scaled ECE: {metrics['scaled_ece']:.4f}, "
          f"Scaled Test Accuracy: {metrics['scaled_test_accuracy']:.2f}%")
    print(f"Class-wise Accuracy: Mean {metrics['class_acc_mean']:.2f}, "
          f"Std {metrics['class_acc_std']:.2f}")
    print(f"Total training time: {(time.time() - start):.2f} seconds")
    
    torch.save(model.state_dict(), config['paths']['final_model'])

if __name__ == "__main__":
    main()