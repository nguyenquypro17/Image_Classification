import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models.Resnet import ResNet20, ResNet56, ResNet110
from models.VGG import VGG16, VGG19
import time 

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
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
    
    return running_loss/len(trainloader), 100.*correct/total

def test(model, testloader, criterion, device):
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
    
    return test_loss/len(testloader), 100.*correct/total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data aussgmentation cho CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Train các models
    models = {
        'ResNet20': ResNet20(),
        'ResNet56': ResNet56(),
        'ResNet110': ResNet110(),
        'VGG16': VGG16(),
        'VGG19': VGG19()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'\n=== Training {name} ===')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
        
        best_acc = 0
        epochs = 160
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc = test(model, testloader, criterion, device)
            scheduler.step()
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'checkpoints/{name}_best.pth')
            
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {time.time()-start_time:.1f}s')
        
        results[name] = best_acc
    
    print('\n=== Final Results ===')
    for name, acc in results.items():
        print(f'{name}: {acc:.2f}%')

if __name__ == '__main__':
    main()
