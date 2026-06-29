import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import VisionDataset
from torchvision.models import resnet18, resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights
import matplotlib.pyplot as plt
from torchvision import transforms
from openpuf.model_relevant import ModifiedResNet18, ModifiedResNet34
from dataset_relevant import PUFdataset
import numpy as np
import random
import os
from tqdm import tqdm
import argparse

torch.set_num_threads(16)


# python ./train_closed_classifier.py --size 220 --num_c 3  --b 16 --device cuda:3

# Argument parser
parser = argparse.ArgumentParser(description='Train ResNet on SRAM PUF dataset')
parser.add_argument('--size', type=int, default=50, help='Dataset size (default: 220)')
parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate (default: 0.0001)')
parser.add_argument('--b', type=int, default=256, help='Batch size (default: 128)')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (default: 15)')
parser.add_argument('--seed', type=int, default=999, help='Random seed (default: 999)')
parser.add_argument('--num_c', type=int, default=45, help='Number of classes (default: 25)')
parser.add_argument('--closed_model', type=str, default='res18', choices=['res18', 'res34'], help='Model architecture (default: res18)')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:3)')
args = parser.parse_args()

# Set parameters from arguments
size = args.size
num_epochs = args.num_epochs
seed = args.seed
base_lr = args.lr
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
batch_size = args.b
num_classes = args.num_c
closed_model = args.closed_model
val_accuracies = []

class_number = f'closed{num_classes}'
ablation_dir = os.path.join(f'/mnt/ssd1/wangxin/dac/experiments_results', class_number)
print(ablation_dir)
closed_weight_dir = os.path.join(ablation_dir,'resnet_weights')
os.makedirs(closed_weight_dir, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.manual_seed(seed)  #99  lr = 0.0005
random.seed(seed)
np.random.seed(seed)

if closed_model == "res34":
    # model_resnet34 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model_resnet18 = resnet34(weights=None)
    modified_resnet = ModifiedResNet34(model=model_resnet34, num_classes=num_classes)
elif closed_model == "res18":
    # model_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model_resnet18 = resnet18(weights=None)
    modified_resnet = ModifiedResNet18(model=model_resnet18, num_classes=num_classes)
else:
    raise ValueError("Invalid closed_model. Choose either 'res34' or 'res18'.")

# Xavier initializations
nn.init.xavier_normal_(modified_resnet.fc.weight) 
nn.init.zeros_(modified_resnet.fc.bias)      

other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

# configure the dataset and dataloader of enrolled devices
closed_trainset = PUFdataset(f'/mnt/ssd1/wangxin/dac/SRAM_split_dataset/size{size}device{num_classes}/closed/train', transform = other_transform)
closed_valset = PUFdataset(f'/mnt/ssd1/wangxin/dac/SRAM_split_dataset/size{size}device{num_classes}/closed/val', transform = other_transform)
closed_testset = PUFdataset(f'/mnt/ssd1/wangxin/dac/SRAM_split_dataset/size{size}device{num_classes}/closed/test', transform = other_transform)
train_loader = DataLoader(closed_trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(closed_valset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(closed_testset, batch_size=batch_size, shuffle=False)

modified_resnet = modified_resnet.to(device)
criterion = nn.CrossEntropyLoss()

feature_extractor_params = []
classifier_params = []
for name, param in modified_resnet.named_parameters():
    if 'fc' not in name: 
        feature_extractor_params.append(param)
    else:
        classifier_params.append(param)

optimizer = optim.Adam([
    {'params': feature_extractor_params, 'lr': base_lr, 'weight_decay': 1e-3}, # Smaller LR for pre-trained backbone
    {'params': classifier_params, 'lr': base_lr, 'weight_decay': 1e-3} # Normal LR for new/fine-tuned head
])

modified_resnet.train()
best_acc = 0.0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(1, num_epochs+1):
    modified_resnet.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs, feat = modified_resnet(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_avg_loss = running_loss / len(train_loader)
    train_losses.append(train_avg_loss) 
    
    # validation during training
    modified_resnet.eval()
    correct = 0
    total = 0
    val_epoch_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = modified_resnet(inputs)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_acc = correct / total
    val_accuracies.append(epoch_acc)
    
    val_avg_loss = val_epoch_loss / len(val_loader)
    val_losses.append(val_avg_loss)
    print(f'Epoch {epoch}/{num_epochs} | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val Acc: {epoch_acc:.4f}')
    
    # Save the best model
    if epoch_acc >= best_acc:
        best_acc = epoch_acc
        best_model_path = os.path.join(closed_weight_dir, f'best_{closed_model}_{num_classes}puf_size{size}.pth')    
        torch.save(modified_resnet.state_dict(), best_model_path)
        print(f"Epoch {epoch} weights saved! Best Validation Acc: {best_acc:.4f}")
        
# Load the weights of the best model and conduct the test
best_model_path = os.path.join(closed_weight_dir, f'best_{closed_model}_{num_classes}puf_size{size}.pth')
modified_resnet.load_state_dict(torch.load(best_model_path))

modified_resnet.eval() 
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, fea = modified_resnet(inputs)s
        
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total
print(f'\nTest set accuracy: {test_acc:.4f}')

# --- drawing ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
s
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(ablation_dir, f'train_metrics_{closed_model}_{num_classes}_size{size}.png'))
plt.show()

print(f"The training indicator graph has been saved to: {os.path.join(ablation_dir, f'train_metrics_{closed_model}_{num_classes}_size{size}.png')}")
