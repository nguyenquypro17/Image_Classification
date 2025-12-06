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
from models.VGG import VGG16

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train 1 epoch"""
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


def get_optimizer(optimizer_name, model_params, lr=0.1):
    """
    Get optimizer based on name
    Args:
        optimizer_name: 'sgd', 'adam', 'adamw', or 'rmsprop'
        model_params: Model parameters
        lr: Learning rate
    
    Returns:
        optimizer instance
    """
    weight_decay = 1e-4
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
    """Plot training and test loss/accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
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


def plot_comparison(results_dict, save_dir='checkpoints'):
    """Plot comparison of 2 models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, metric in enumerate(metrics):
        models = list(results_dict.keys())
        values = [results_dict[m].get(metric, 0) for m in models]
        
        bars = axes[idx].bar(models, values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 105])
        
        for bar, v in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f'Comparison plot saved to {plot_path}')
    plt.close()


def train_model(model, model_name, trainloader, testloader, device, 
                epochs=160, lr=0.1, optimizer_name='sgd', save_dir='checkpoints',
                weight_decay=1e-4, batch_size=128):
    """Train một model hoàn chỉnh"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    best_acc = 0.0
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    print(f'\n{"="*80}')
    print(f'Model: {model_name:15s} | Optimizer: {optimizer_name.upper():8s} | LR: {lr} | Batch: {batch_size}')
    print(f'Weight Decay: {weight_decay} | Epochs: {epochs}')
    print(f'{"="*80}')
    
    start_total_time = time.time()
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, testloader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.4f} | '
                  f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | '
                  f'Best: {best_acc:.2f}%')
    
    total_time = time.time() - start_total_time
    print(f'\nTraining finished! Best accuracy: {best_acc:.2f}% | Total time: {total_time/60:.2f} minutes')
    
    # Plot training curves
    plot_training_curves(history, model_name, save_dir)
    
    # Evaluate metrics
    metrics = evaluate_metrics(model, testloader, device)
    
    return {
        'best_acc': best_acc,
        'total_time': total_time,
        'history': history,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train classification models on CIFAR-10')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--augmentation', type=str, default='standard',
                        help='Augmentation type')
    parser.add_argument('--epochs', type=int, default=160,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--model', type=str, default=None,
                        choices=['ResNet20', 'ResNet56', 'VGG16', 'all'],
                        help='Model to train: ResNet20, ResNet56, VGG16, or all')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare 2 models (e.g., ResNet20,VGG16)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    # Load data
    print('\nLoading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_loaders(
        batch_size=args.batch_size,
        augmentation=args.augmentation
    )
    print(f'Train batches: {len(trainloader)}, Test batches: {len(testloader)}')
    
    # Định nghĩa các models
    models_dict = {
        'ResNet20': ResNet20(),
        'ResNet56': ResNet56(),
        'VGG16': VGG16()
    }
    
    results = {}
    
    if args.compare:
        # Compare 2 models
        compare_models = [m.strip() for m in args.compare.split(',')]
        for model_name in compare_models:
            if model_name not in models_dict:
                print(f'Model {model_name} not found!')
                continue
            
            model = models_dict[model_name].to(device)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'\n{model_name} parameters: {num_params/1e6:.2f}M')
            
            result = train_model(
                model=model,
                model_name=model_name,
                trainloader=trainloader,
                testloader=testloader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                optimizer_name=args.optimizer,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size
            )
            results[model_name] = result['metrics']
        
        # Plot comparison
        if len(results) == 2:
            plot_comparison(results)
    elif args.model and args.model != 'all':
        # Train single model
        model_name = args.model
        if model_name not in models_dict:
            print(f'Model {model_name} not found!')
            return
        
        model = models_dict[model_name].to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'\n{model_name} parameters: {num_params/1e6:.2f}M')
        
        result = train_model(
            model=model,
            model_name=model_name,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )
        results[model_name] = result
    else:
        # Train all models
        for name, model in models_dict.items():
            model = model.to(device)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'\n{name} parameters: {num_params/1e6:.2f}M')
            
            result = train_model(
                model=model,
                model_name=name,
                trainloader=trainloader,
                testloader=testloader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                optimizer_name=args.optimizer,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size
            )
            results[name] = result
    
    # In kết quả cuối cùng
    print('\n' + '='*80)
    print('FINAL RESULTS')
    print('='*80)
    for name, result in results.items():
        if isinstance(result, dict) and 'accuracy' in result:
            print(f'{name:15s}: Acc={result["accuracy"]:.2f}% | F1={result["f1"]:.2f}% | '
                  f'Prec={result["precision"]:.2f}% | Rec={result["recall"]:.2f}%')
        elif isinstance(result, dict) and 'best_acc' in result:
            print(f'{name:15s}: Best Acc={result["best_acc"]:.2f}% | Time={result["total_time"]/60:.2f}min | '
                  f'Accuracy={result["metrics"]["accuracy"]:.2f}% | F1={result["metrics"]["f1"]:.2f}%')
    print('='*80)


if __name__ == '__main__':
    main()