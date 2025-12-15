import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.data_loader import get_cifar10_loaders
from models.Resnet import ResNet20, ResNet56, ResNet110
from models.VGG import VGG16, VGG19 

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train 1 epoch trả về train loss và train accuracy"""
    model.train()
    running_loss = 0.0 
    correct = 0 
    total = 0 
    
    for inputs, targets in trainloader: # batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc


def test_epoch(model, testloader, criterion, device):
    """Test model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100.0 * correct / total
    
    return test_loss, test_acc


def val_epoch(model, valloader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = val_loss / len(valloader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc


def get_optimizer(optimizer_name, model_params, lr=0.1, weight_decay=1e-4):
    """
    Get optimizer based on name
    Args:
        optimizer_name: 'sgd', 'adam', 'adamw', or 'rmsprop'
        model_params: Model parameters
        lr: Learning rate
        weight_decay: Weight decay
    
    Returns:
        optimizer instance
    """
    momentum = 0.9
    
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


def evaluate_metrics(model, testloader, device):
    """Calculate Accuracy, F1, Precision, Recall"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds) * 100,
        'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
        'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
        'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
    }
    return metrics


def plot_training_curves(history, model_name, save_dir='checkpoints'):
    """Plot training and validation loss/accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curve', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_training_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f'Training curve saved to {plot_path}')
    plt.close()


def train_model(model, model_name, trainloader, valloader, testloader, device, 
                epochs=160, lr=0.1, optimizer_name='sgd', save_dir='/kaggle/working/checkpoints',
                weight_decay=1e-4, batch_size=128):
    """Train một model hoàn chỉnh"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Tạo tên subfolder dựa vào cấu hình
    config_name = f'{optimizer_name}_lr{lr}_wd{weight_decay}'
    config_dir = os.path.join(save_dir, model_name, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_total_time = time.time()
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, valloader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        # Tracking best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Bật lại tính năng lưu model (Quan trọng)
            torch.save(model.state_dict(), os.path.join(config_dir, f'{model_name}_best.pth'))
        
        current_lr = optimizer.param_groups[0]['lr']
        
    total_time = time.time() - start_total_time
    print(f'\nTraining finished!')
    
    # Plot training curves
    plot_training_curves(history, model_name, config_dir)
    
    # Test on test set once with best model
    test_loss, test_acc = test_epoch(model, testloader, criterion, device)
    
    # Evaluate metrics
    metrics = evaluate_metrics(model, testloader, device)
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_time': total_time,
        'history': history,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train classification models on CIFAR-10')
    parser.add_argument('--device', type=str, required=True,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--augmentation', type=str, required=True,
                        help='Augmentation type: none, aug1, aug2, etc.')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size')
    parser.add_argument('--weight_decay', type=float, required=True,
                        help='Weight decay')
    # Cập nhật danh sách choices cho argument model
    parser.add_argument('--model', type=str, required=True,
                        choices=['ResNet20', 'ResNet56', 'ResNet110', 'VGG16', 'VGG19'],
                        help='Model to train: ResNet20, ResNet56, ResNet110, VGG16, VGG19')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints',
                        help='Directory to save checkpoints and results')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load data
    trainloader, valloader, testloader = get_cifar10_loaders(
        batch_size=args.batch_size,
        augmentation=args.augmentation
    )
    
    # Cập nhật Dictionary chứa các Model
    models_dict = {
        'ResNet20': ResNet20(),
        'ResNet56': ResNet56(),
        'ResNet110': ResNet110(), # Đã thêm
        'VGG16': VGG16(),
        'VGG19': VGG19()          # Đã thêm
    }
    
    results = {}
    
    # Train single model
    model_name = args.model
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} chưa được định nghĩa trong models_dict!")

    model = models_dict[model_name].to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{model_name} parameters: {num_params/1e6:.2f}M')
    
    result = train_model(
        model=model,
        model_name=model_name,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    results[model_name] = result
    
    print('='*80)
    for name, result in results.items():
        print(f'Accuracy: {result["test_acc"]:.2f}% | F1: {result["metrics"]["f1"]:.2f}% | Precision: {result["metrics"]["precision"]:.2f}% | Recall: {result["metrics"]["recall"]:.2f}% | Time: {result["total_time"]/60:.2f}min')
    print('='*80)


if __name__ == '__main__':
    main()