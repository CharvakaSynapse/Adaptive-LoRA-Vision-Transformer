import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DeiTForImageClassification
from peft import get_peft_model
import matplotlib.pyplot as plt

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy, np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_logits)

def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def find_optimal_temperature(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            all_logits.append(model(images).logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    temperatures = np.linspace(0.2, 3.0, 100)
    best_temp, min_ece = 1.0, float('inf')
    for t in temperatures:
        scaled_probs = F.softmax(all_logits / t, dim=1).numpy()
        ece = compute_ece(scaled_probs, all_labels)
        if ece < min_ece:
            min_ece, best_temp = ece, t
    return best_temp

def evaluate_model(model, test_loader, val_loader, device, best_model_path, adaptive_ranks):
    base_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')
    base_model.classifier = torch.nn.Linear(base_model.classifier.in_features, 100)
    lora_config = model.peft_config['default']
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    for name, rank in adaptive_ranks.items():
        module = model.get_submodule(name)
        if hasattr(module, 'r'):
            if isinstance(module.r, dict) and 'default' in module.r:
                module.r['default'] = rank
            else:
                module.r = rank
            in_features = module.in_features
            out_features = module.out_features
            module.lora_A['default'].weight.data = torch.randn(rank, in_features).to(device) * 0.02
            module.lora_B['default'].weight.data = torch.zeros(out_features, rank).to(device)
    
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss, test_accuracy, test_probs, test_labels, test_logits = validate(model, test_loader, criterion, device)
    ece = compute_ece(test_probs, test_labels)
    
    class_accuracies = []
    for i in range(100):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(test_labels[class_mask], np.argmax(test_probs[class_mask], axis=1))
            class_accuracies.append(class_acc)
    class_acc_mean = np.mean(class_accuracies)
    class_acc_std = np.std(class_accuracies)
    
    optimal_temp = find_optimal_temperature(model, val_loader, device)
    scaled_test_probs = F.softmax(torch.tensor(test_logits) / optimal_temp, dim=1).numpy()
    scaled_test_accuracy = accuracy_score(test_labels, np.argmax(scaled_test_probs, axis=1)) * 100
    scaled_ece = compute_ece(scaled_test_probs, test_labels)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'ece': ece,
        'scaled_ece': scaled_ece,
        'scaled_test_accuracy': scaled_test_accuracy,
        'class_acc_mean': class_acc_mean,
        'class_acc_std': class_acc_std
    }

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, config):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config['paths']['plot_output'])
    plt.close()