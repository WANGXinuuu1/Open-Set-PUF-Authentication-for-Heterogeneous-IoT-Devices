import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import blitz.modules as bnn
import blitz.losses as bl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset_relevant import PUFdataset
from pytorch_ood.utils import ToUnknown
from model_relevant import BCNN
import argparse


# Argument parser
parser = argparse.ArgumentParser(description='Train BCNN on SRAM PUF dataset')
parser.add_argument('--exp', type=int, default=99, help='Experiment number (default: 999)')
parser.add_argument('--size', type=int, default=220, help='Dataset size 220X200)')
parser.add_argument('--num_d', type=int, default=3, help='Number of classes (default: 40)')
parser.add_argument('--seed', type=int, default=88, help='Random seed (default: 88)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)')
parser.add_argument('--b', type=int, default=32, help='Batch size (default: 128)')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs (default: 60)')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5)')
parser.add_argument('--mc_samples', type=int, default=20, help='Monte Carlo sampling times for inference (default: 20)')
parser.add_argument('--run_train', action='store_true', help='Run training phase')
parser.add_argument('--run_val', action='store_true', help='Run validation phase')
parser.add_argument('--run_test', action='store_true', help='Run test phase')
args = parser.parse_args()

# If no phase is specified, run all phases by default
if not (args.run_train or args.run_val or args.run_test):
    args.run_train = True
    args.run_val = True
    args.run_test = True

# Set parameters from arguments
size = args.size
num_classes = args.num_d
seed = args.seed
lr = args.lr
beta1 = args.beta1
batch_size = args.b
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
num_epochs = args.num_epochs
dropout_rate = args.dropout
mc_samples = args.mc_samples
exp = args.exp

print(f"Using device: {device}")
print(f"Running phases: Train={args.run_train}, Validation={args.run_val}, Test={args.run_test}")

# Create unique directory name based on hyperparameters
class_number = f'closed{num_classes}'
params_dir = f'exp{exp}bcnn'
ablation_dir = os.path.join(f'./bcnn_experiments', class_number, params_dir)

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


data_transform = transforms.Compose([
    transforms.ToTensor()])

if args.run_train:
    train_closedset = PUFdataset(
        f'./DRAM_dataset/size{size}device{num_classes}/closed/train',
        transform=data_transform
    )
    train_closed_loader = DataLoader(train_closedset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = BCNN(num_classes=num_classes)
    model.to(device)

    # Loss function and optimizer
    kl_weight = 1e-3  
    criterion = nn.CrossEntropyLoss(reduction='mean') 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_results_dir = os.path.join(ablation_dir, 'train')
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Training results save path: {save_results_dir}")

    weight_save_dir = os.path.join(ablation_dir, 'bcnn_weights')
    os.makedirs(weight_save_dir, exist_ok=True)
    print(f"Weight save path: {weight_save_dir}")

    # Training loop
    train_losses = []
    train_accuracies = []

    print("\n--- Starting BCNN Training ---")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_closed_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            likelihood_loss = criterion(outputs, labels)
            kl_divergence = bl.kl_divergence_from_nn(model)
            loss = likelihood_loss + kl_weight * kl_divergence
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_closed_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save model checkpoint
        cur_model_wts = copy.deepcopy(model.state_dict())
        path_to_save_paramOnly = os.path.join(weight_save_dir, f'epoch-{epoch}.pth')
        torch.save(cur_model_wts, path_to_save_paramOnly)
        print(f"Model saved to {path_to_save_paramOnly}")

    plt.figure(figsize=(10, 5))
    plt.title("BCNN Training Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_results_dir, 'training_loss.png'), bbox_inches='tight')
    plt.close()
    # Plot training accuracy
    plt.figure(figsize=(10, 5))
    plt.title("BCNN Training Accuracy")
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(save_results_dir, 'training_accuracy.png'), bbox_inches='tight')
    plt.close()


#### Validation phase ####
if args.run_val:
    weight_save_dir = os.path.join(ablation_dir, 'bcnn_weights')
    save_results_dir = os.path.join(ablation_dir, 'validate')
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Validation results save path: {save_results_dir}")
    val_closedset = PUFdataset(
        f'./DRAM_dataset/size{size}device{num_classes}/closed/val',
        transform=data_transform
    )
    val_openset = PUFdataset(
        f'./DRAM_dataset/size{size}device{num_classes}/val_out',
        transform=data_transform,
        target_transform=ToUnknown()
    )
    validation_loader = DataLoader(
        ConcatDataset([val_closedset, val_openset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model = BCNN(num_classes=num_classes)
    model.to(device)
    closed_acc_list = []
    far_list = []
    frr_list = []

    epoch = 50
    weights_path = os.path.join(weight_save_dir, f"epoch-{epoch}.pth")
    if not os.path.exists(weights_path):
        print(f"Weights file for epoch {epoch} not found, skipping.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    correct_closed_pure = 0
    false_accept_total = 0
    false_reject_total = 0
    total_open_samples = 0
    total_closed_samples = 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            # Monte Carlo sampling for uncertainty estimation
            monte_carlo_logits_for_batch = []
            for _ in range(mc_samples):
                model.train()  # Temporarily enable randomness
                logits_sample = model(data)
                model.eval()   # Restore evaluation mode
                monte_carlo_logits_for_batch.append(logits_sample.cpu().numpy())
            
            monte_carlo_logits_for_batch = np.stack(monte_carlo_logits_for_batch)
            monte_carlo_probabilities_for_batch = torch.softmax(torch.from_numpy(monte_carlo_logits_for_batch), dim=-1).numpy()
            mean_prob = monte_carlo_probabilities_for_batch.mean(axis=0) # for closed-set acc
            # Calculate P10 and P90 percentiles
            p10_per_class = np.percentile(monte_carlo_probabilities_for_batch, 10, axis=0)
            p90_per_class = np.percentile(monte_carlo_probabilities_for_batch, 90, axis=0)
            # Separate closed and open samples
            closed_mask = (target != -1)
            open_mask = (target == -1)

            # Update totals
            total_closed_samples += closed_mask.sum().item()
            total_open_samples += open_mask.sum().item()

            # Process each sample in the batch
            for sample_idx_in_batch in range(data.shape[0]):
                true_label = target[sample_idx_in_batch].item()
                # Get candidate predicted class
                candidate_known_class_prediction = np.argmax(p90_per_class[sample_idx_in_batch])
                # Get P10 value for candidate class
                p10_for_candidate_class = p10_per_class[sample_idx_in_batch, candidate_known_class_prediction]
                # Get max P90 of other classes
                other_known_classes_indices = [j for j in range(num_classes) if j != candidate_known_class_prediction]
                max_p90_of_other_classes = np.max(p90_per_class[sample_idx_in_batch, other_known_classes_indices])
                # Decision rule
                if (p90_per_class[sample_idx_in_batch, candidate_known_class_prediction] > 0.95) and (p10_for_candidate_class >= max_p90_of_other_classes):
                    predicted_label = candidate_known_class_prediction
                else:
                    predicted_label = -1
                # Count metrics
                if true_label != -1:  # Closed-set sample
                    pure_pred = int(np.argmax(mean_prob[sample_idx_in_batch]))
                    if pure_pred == true_label:
                        correct_closed_pure += 1
                    if predicted_label == -1:
                        false_reject_total += 1
                else:  # Open-set sample
                    if predicted_label != -1:
                        false_accept_total += 1
        
        # Calculate metrics
        closed_acc = 100 * correct_closed_pure / total_closed_samples if total_closed_samples > 0 else 0
        far = 100 * false_accept_total / total_open_samples if total_open_samples > 0 else 0
        frr = 100 * false_reject_total / total_closed_samples if total_closed_samples > 0 else 0
        closed_acc_list.append(closed_acc)
        far_list.append(far)
        frr_list.append(frr)

        print(f"\n--- Validation Results for epoch {epoch} ---")
        print(f"Total closed-set samples: {total_closed_samples}")
        print(f"Total open-set samples: {total_open_samples}")
        print(f"Closed-set Accuracy: {closed_acc:.2f}%")
        print(f"FAR (False Acceptance Rate): {far:.2f}%")
        print(f"FRR (False Rejection Rate): {frr:.2f}%")
        print(f"Monte Carlo Sampling Times: {mc_samples}")
        result_str = f"Val epoch {epoch}: closed Acc={closed_acc:.2f}%, FAR={far:.2f}%, FRR={frr:.2f}%"


#### Test phase ####
if args.run_test:
    weight_save_dir = os.path.join(ablation_dir, 'bcnn_weights')
    save_results_dir = os.path.join(ablation_dir, 'test')
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Test results save path: {save_results_dir}")

    # Define test datasets
    test_closedset = PUFdataset(
        f'./DRAM_dataset/size{size}device{num_classes}/closed/test',
        transform=data_transform
    )
    test_openset = PUFdataset(
        f'./DRAM_dataset/size{size}device{num_classes}/test_out',
        transform=data_transform,
        target_transform=ToUnknown()
    )
    test_loader = DataLoader(
        ConcatDataset([test_closedset, test_openset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = BCNN(num_classes=num_classes)
    model.to(device)
    # Lists to store results
    closed_acc_list = []
    far_list = []
    frr_list = []

    epoch = 50
    weights_path = os.path.join(weight_save_dir, f"epoch-{epoch}.pth")
    if not os.path.exists(weights_path):
        print(f"Weights file for epoch {epoch} not found, skipping.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    correct_closed_pure = 0
    false_accept_total = 0
    false_reject_total = 0
    total_open_samples = 0
    total_closed_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Monte Carlo sampling for uncertainty estimation
            monte_carlo_logits_for_batch = []
            for _ in range(mc_samples):
                model.train()  # Temporarily enable randomness
                logits_sample = model(data)
                model.eval()   # Restore evaluation mode
                monte_carlo_logits_for_batch.append(logits_sample.cpu().numpy())
            
            monte_carlo_logits_for_batch = np.stack(monte_carlo_logits_for_batch)
            monte_carlo_probabilities_for_batch = torch.softmax(torch.from_numpy(monte_carlo_logits_for_batch), dim=-1).numpy()
            mean_prob = monte_carlo_probabilities_for_batch.mean(axis=0) # for closed-set acc
            # Calculate P10 and P90 percentiles
            p10_per_class = np.percentile(monte_carlo_probabilities_for_batch, 10, axis=0)
            p90_per_class = np.percentile(monte_carlo_probabilities_for_batch, 90, axis=0)

            # Separate closed and open samples
            closed_mask = (target != -1)
            open_mask = (target == -1)

            # Update totals
            total_closed_samples += closed_mask.sum().item()
            total_open_samples += open_mask.sum().item()

            # Process each sample in the batch
            for sample_idx_in_batch in range(data.shape[0]):
                true_label = target[sample_idx_in_batch].item()
                
                # Get candidate predicted class
                candidate_known_class_prediction = np.argmax(p90_per_class[sample_idx_in_batch])
                
                # Get P10 value for candidate class
                p10_for_candidate_class = p10_per_class[sample_idx_in_batch, candidate_known_class_prediction]
                
                # Get max P90 of other classes
                other_known_classes_indices = [j for j in range(num_classes) if j != candidate_known_class_prediction]
                max_p90_of_other_classes = np.max(p90_per_class[sample_idx_in_batch, other_known_classes_indices])
                
                # Decision rule
                if (p90_per_class[sample_idx_in_batch, candidate_known_class_prediction] > 0.95) and (p10_for_candidate_class >= max_p90_of_other_classes):
                    predicted_label = candidate_known_class_prediction
                else:
                    predicted_label = -1
                
                # Count metrics
                if true_label != -1:  # Closed-set sample
                    pure_pred = int(np.argmax(mean_prob[sample_idx_in_batch]))
                    if pure_pred == true_label:
                        correct_closed_pure += 1
                    if predicted_label == -1:
                        false_reject_total += 1
                else:  # Open-set sample
                    if predicted_label != -1:
                        false_accept_total += 1
    
    # Calculate metrics
    closed_acc = 100 * correct_closed_pure / total_closed_samples if total_closed_samples > 0 else 0
    far = 100 * false_accept_total / total_open_samples if total_open_samples > 0 else 0
    frr = 100 * false_reject_total / total_closed_samples if total_closed_samples > 0 else 0

    closed_acc_list.append(closed_acc)
    far_list.append(far)
    frr_list.append(frr)

    print(f"\n--- Test Results for epoch {epoch} ---")
    print(f"Total closed-set samples: {total_closed_samples}")
    print(f"Total open-set samples: {total_open_samples}")
    print(f"Closed-set Accuracy: {closed_acc:.2f}%")
    print(f"FAR (False Acceptance Rate): {far:.2f}%")
    print(f"FRR (False Rejection Rate): {frr:.2f}%")
    print(f"Monte Carlo Sampling Times: {mc_samples}")
    result_str = f"Test epoch {epoch}: closed Acc={closed_acc:.2f}%, FAR={far:.2f}%, FRR={frr:.2f}%"