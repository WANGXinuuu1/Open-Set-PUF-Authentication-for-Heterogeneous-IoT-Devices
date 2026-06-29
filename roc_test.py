import os, random, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet18, resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from model_relevant import ModifiedResNet18, Generator, Discriminator, ModifiedResNet34
from dataset_relevant import PUFdataset 
from pytorch_ood.utils import ToUnknown
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, roc_curve
import matplotlib.font_manager as fm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate OpenGAN on SRAM PUF dataset')
parser.add_argument('--exp', type=int, default=999, help='Experiment number (default: 999)')
parser.add_argument('--size', type=int, default=50, help='Dataset size (default: 220)')
parser.add_argument('--num_d', type=int, default=40, help='Number of classes (default: 25)')
parser.add_argument('--seed', type=int, default=99, help='Random seed (default: 99)')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
parser.add_argument('--ngf', type=int, default=64, help='Generator feature map size (default: 256)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)')
parser.add_argument('--b', type=int, default=128, help='Batch size (default: 256)')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use (default: cuda:0)')
parser.add_argument('--run_val', action='store_true', help='Run validation phase')
parser.add_argument('--run_test', action='store_true', help='Run test phase')
parser.add_argument('--epoch', type=int, required=True, help='Specific epoch of discriminator weights to load')
args = parser.parse_args()

# If no phase is specified, run both validation and test by default
if not (args.run_val or args.run_test):
    args.run_val = True
    args.run_test = True

# Set parameters from arguments
size = args.size
num_classes = args.num_d
seed = args.seed
lr = args.lr
ndf = ngf = args.ngf
beta1 = args.beta1
batch_size = args.b
epoch = args.epoch
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

exp = args.exp
closed_model = 'res18'
nc = 512
nz = 100
threshold = 0.5

class_number = f'closed{num_classes}'

# Create unique directory path based on hyperparameter settings
params_dir = f'exp{exp}results_size{size}_numd{num_classes}_seed{seed}_lr{lr}_ngf{ngf}_beta1{beta1}_b{batch_size}'
ablation_dir = os.path.join(f'./experiments_results', class_number, params_dir)
weight_save_dir = os.path.join(ablation_dir, 'gan_weights')
weights_path = os.path.join(weight_save_dir, f"epoch-{epoch}.DNet")

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

closed_weight_dir = os.path.join(f'./experiments_results', class_number, 'resnet_weights')
closed_weight_path = os.path.join(closed_weight_dir, f'best_{closed_model}_{num_classes}puf_size{size}.pth')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.manual_seed(seed)  #99  lr = 0.0005
random.seed(seed)
np.random.seed(seed)

other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

if closed_model == "res18":
    model_resnet18 = resnet18(weights=None)
    modified_resnet = ModifiedResNet18(model=model_resnet18, num_classes=num_classes)
elif closed_model == "res34":
    model_resnet34 = resnet34(weights=None)
    modified_resnet = ModifiedResNet34(model=model_resnet34, num_classes=num_classes)
else:
    raise ValueError("Invalid closed_model. Choose either 'res34' or 'res18'.")

modified_resnet.load_state_dict(torch.load(closed_weight_path, weights_only=True))
modified_resnet.to(device)
modified_resnet.eval()


#### Validation phase ####
if args.run_val:
    save_results_dir = os.path.join(ablation_dir, 'validate')
    os.makedirs(save_results_dir, exist_ok=True)

    # Configure dataset and dataloader
    # Masks for enrolled and outlier samples respectively
    val_closedset = PUFdataset(f'./SRAM_split_dataset/size{size}device{num_classes}/closed/val', transform = other_transform)
    val_openset = PUFdataset(f"./SRAM_split_dataset/size{size}device{num_classes}/val_out", transform = other_transform, target_transform=ToUnknown())
    validation_loader = DataLoader(ConcatDataset([val_closedset, val_openset]), batch_size=batch_size, shuffle=False, num_workers=4)

    netD = Discriminator(nc=nc, ndf=ndf).to(device)
    netD.load_state_dict(torch.load(weights_path, map_location=device))
    netD.eval()

    # Open file to save validation results
    val_results_file = os.path.join(save_results_dir, f'validate_epoch{epoch}_roc.txt')
    val_file = open(val_results_file, 'w')
    val_file.write("Validation Results with ROC Metrics\n")
    val_file.write("="*80 + "\n")
    val_file.write(f"Experiment: {exp}, Epoch: {epoch}\n")
    val_file.write("="*80 + "\n\n")

    # Initialize the counter
    correct_open_total = 0
    correct_closed_total = 0
    total_open_samples = 0
    total_closed_samples = 0
    
    # Collect all predicted scores and true labels for ROC calculation
    all_disc_scores = []
    all_binary_labels = []  
    closed_set_scores = []  
    open_set_scores = [] 
    
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            
            closed_result, features = modified_resnet(data)
            _, closed_predicted = torch.max(closed_result.data, 1)
            
            # Get discriminator output and ensure correct shape
            disc_output = netD(features)
            if disc_output.dim() > 1:
                disc_output = disc_output.view(disc_output.size(0), -1).squeeze(-1)
            else:
                disc_output = disc_output.squeeze()
            
            binary_predictions = (disc_output > threshold)
            
            # Masks for enrolled and outlier samples respectively
            closed_mask = (target != -1)
            open_mask = (target == -1)
            
            all_disc_scores.extend(disc_output.cpu().numpy().tolist())
            binary_labels = torch.zeros_like(target, dtype=torch.float)
            binary_labels[closed_mask] = 1.0  # closed-set = 1
            all_binary_labels.extend(binary_labels.cpu().numpy().tolist())
            
            if closed_mask.any():
                closed_set_scores.extend(disc_output[closed_mask].cpu().numpy().tolist())
            if open_mask.any():
                open_set_scores.extend(disc_output[open_mask].cpu().numpy().tolist())
            
            # Calculate the correctly authenticated enrolled samples
            correct_closed = ((binary_predictions[closed_mask] == True) & 
                            (closed_predicted[closed_mask] == target[closed_mask])).sum().item()
            
            # Calculate the successfully rejected outlier samples (rejected by the discriminator)
            correct_open = (binary_predictions[open_mask] == False).sum().item()
            
            total_closed_samples += closed_mask.sum().item()
            total_open_samples += open_mask.sum().item()
            correct_closed_total += correct_closed
            correct_open_total += correct_open
    
    correct_enrolled_rate = correct_closed_total / total_closed_samples if total_closed_samples > 0 else 0
    correct_open_rate = correct_open_total / total_open_samples if total_open_samples > 0 else 0
    
    all_disc_scores = np.array(all_disc_scores)
    all_binary_labels = np.array(all_binary_labels)
    
    auroc = roc_auc_score(all_binary_labels, all_disc_scores)
    
    # calculate Precision-Recall curve and AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(all_binary_labels, all_disc_scores)
    auprc = auc(recall_curve, precision_curve)
    
    # Calculate the binary classification index based on the threshold
    binary_preds = (all_disc_scores > threshold).astype(int)
    f1 = f1_score(all_binary_labels, binary_preds)
    precision = precision_score(all_binary_labels, binary_preds)
    recall = recall_score(all_binary_labels, binary_preds)
    

    result_str = f"Val Epoch {epoch}: Enrolled={correct_enrolled_rate * 100:.2f}%, Open={correct_open_rate * 100:.2f}%, AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
    
    print(result_str)
    val_file.write(result_str + "\n")
    val_file.close()

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(6.5, 5))

    bins = np.linspace(min(min(closed_set_scores), min(open_set_scores)), 
                       max(max(closed_set_scores), max(open_set_scores)), 40)
    
    ax.hist(open_set_scores, bins=bins, alpha=0.7, label='Outlier devices', 
            color='#FF6B6B', edgecolor='white', linewidth=0.5, density=True)
    ax.hist(closed_set_scores, bins=bins, alpha=0.7, label='Enrolled devices', 
            color='#4ECDC4', edgecolor='white', linewidth=0.5, density=True)
    
    # set threshould line
    ax.axvline(x=threshold, color='#2C3E50', linestyle='--', linewidth=2.5, 
               label=f'Decision threshold ({threshold})', zorder=10)

    ax.set_xlabel('Discriminator Output Confidence $P_{open}$', fontsize=22, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=22, fontweight='bold')
    #ax.set_title('Distribution of Discriminator Outputs for Enrolled and Outlier Samples', fontsize=18, fontweight='bold', pad=15)

    ax.tick_params(axis='both', which='major', labelsize=16, direction='in', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)

    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, which='major')
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=13, frameon=True, shadow=True, fancybox=True, 
                       loc='upper right', framealpha=0.95, edgecolor='gray')
    legend.get_frame().set_linewidth(1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor('#2C3E50')

    textstr = f'Enrolled: $\mu$={np.mean(closed_set_scores):.3f}, $\sigma$={np.std(closed_set_scores):.3f}\n'
    textstr += f'Outlier: $\mu$={np.mean(open_set_scores):.3f}, $\sigma$={np.std(open_set_scores):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.15, edgecolor='gray', linewidth=1)
    ax.text(0.98, 0.72, textstr, transform=ax.transAxes, fontsize=13,
            horizontalalignment='right', verticalalignment='top', bbox=props, family='serif')
    
    plt.tight_layout()
    distribution_path = os.path.join(save_results_dir, 'distribution.pdf')
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Distribution plot saved to: {distribution_path}")   

#### Test phase ####
if args.run_test:

    save_results_dir = os.path.join(ablation_dir, 'test')
    os.makedirs(save_results_dir, exist_ok=True)

    test_closedset = PUFdataset(f'./SRAM_split_dataset/size{size}device{num_classes}/closed/test', 
                                transform = other_transform)
    test_openset = PUFdataset(f"./SRAM_split_dataset/size{size}device{num_classes}/test_out", 
                              transform = other_transform, target_transform=ToUnknown())
    test_loader = DataLoader(ConcatDataset([test_closedset, test_openset]), batch_size=batch_size, shuffle=False, num_workers=4)

    netD = Discriminator(nc=nc, ndf=ndf).to(device)
    netD.load_state_dict(torch.load(weights_path, map_location=device))
    netD.eval()

    # Open file to save test results
    test_results_file = os.path.join(save_results_dir, f'test_epoch{epoch}_roc.txt')
    test_file = open(test_results_file, 'w')
    test_file.write("Test Results with ROC Metrics\n")
    test_file.write("="*80 + "\n")
    test_file.write(f"Experiment: {exp}, Epoch: {epoch}\n")
    test_file.write(f"Params: size={size}, num_d={num_classes}, seed={seed}, lr={lr}, ngf={ngf}, beta1={beta1}, batch={batch_size}\n")
    test_file.write("="*80 + "\n\n")

    correct_open_total = 0
    correct_closed_total = 0
    total_open_samples = 0
    total_closed_samples = 0

    all_disc_scores = []
    all_binary_labels = []  # 1 for closed-set, 0 for open-set
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            closed_result, features = modified_resnet(data)
            _, closed_predicted = torch.max(closed_result.data, 1)
            
            # Get discriminator output and ensure correct shape
            disc_output = netD(features)
            if disc_output.dim() > 1:
                disc_output = disc_output.view(disc_output.size(0), -1).squeeze(-1)
            else:
                disc_output = disc_output.squeeze()
            
            binary_predictions = (disc_output > threshold)

            closed_mask = (target != -1)
            open_mask = (target == -1)
            

            all_disc_scores.extend(disc_output.cpu().numpy().tolist())
            binary_labels = torch.zeros_like(target, dtype=torch.float)
            binary_labels[closed_mask] = 1.0  # closed-set = 1
            all_binary_labels.extend(binary_labels.cpu().numpy().tolist())
            

            correct_closed = ((binary_predictions[closed_mask] == True) & 
                            (closed_predicted[closed_mask] == target[closed_mask])).sum().item()
            
            correct_open = (binary_predictions[open_mask] == False).sum().item()

            total_closed_samples += closed_mask.sum().item()
            total_open_samples += open_mask.sum().item()
            correct_closed_total += correct_closed
            correct_open_total += correct_open
    

    correct_enrolled_rate = correct_closed_total / total_closed_samples if total_closed_samples > 0 else 0
    correct_open_rate = correct_open_total / total_open_samples if total_open_samples > 0 else 0
    
    all_disc_scores = np.array(all_disc_scores)
    all_binary_labels = np.array(all_binary_labels)
    
    # Calculate the ROC metrics
    auroc = roc_auc_score(all_binary_labels, all_disc_scores)
    
    # Calculate Precision-Recall curve and AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(all_binary_labels, all_disc_scores)
    auprc = auc(recall_curve, precision_curve)
    
    # Calculate the binary classification index based on the threshold
    binary_preds = (all_disc_scores > threshold).astype(int)
    f1 = f1_score(all_binary_labels, binary_preds)
    precision = precision_score(all_binary_labels, binary_preds)
    recall = recall_score(all_binary_labels, binary_preds)
    
    test_result_str = f"Test Epoch {epoch}: Enrolled={correct_enrolled_rate * 100:.2f}%, Open={correct_open_rate * 100:.2f}%, AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
    
    print(test_result_str)
    test_file.write(test_result_str + "\n")
    test_file.close()
