import os, random, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet18, resnet34
# from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from model_relevant import ModifiedResNet18, Generator, Discriminator, ModifiedResNet34
from dataset_relevant import PUFdataset 
from pytorch_ood.utils import ToUnknown
import numpy as np
import argparse

# Initialization for Generator and Discriminator during training phrase
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Argument parser
parser = argparse.ArgumentParser(description='Train OpenGAN on SRAM PUF dataset')
parser.add_argument('--exp', type=int, default=0, help='Experiment number')
parser.add_argument('--threshold', type=float, default=0.5, help='Discriminator threshold (default: 0.5)')
parser.add_argument('--size', type=int, default=50, help='Dataset size (default: 50)')
parser.add_argument('--num_d', type=int, default=40, help='Number of classes')
parser.add_argument('--seed', type=int, default=99, help='Random seed (default: 99)')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
parser.add_argument('--ngf', type=int, default=64, help='Generator feature map size (default: 256)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)')
parser.add_argument('--b', type=int, default=128, help='Batch size (default: 256)')
parser.add_argument('--lamba0', type=float, default=1.0, help='Lambda for real training loss (default: 1.0)')
parser.add_argument('--lamba1', type=float, default=1.0, help='Lambda for generated training loss (default: 1.0)')
parser.add_argument('--device', type=str, default='cuda:3', help='GPU Device(default: cuda:0)')
parser.add_argument('--start_epochs', type=int, default=1, help='Start epoch for validation (default: 1)')
parser.add_argument('--end_epochs', type=int, default=60, help='End epoch for validation (default: 50)')
parser.add_argument('--step', type=int, default=1, help='Step size for epoch validation (default: 1)')
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
ndf = ngf = args.ngf
beta1 = args.beta1
batch_size = args.b
lamba0 = args.lamba0
lamba1 = args.lamba1
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

exp = args.exp
closed_model = 'res18'
num_epochs = args.end_epochs
nc = 512
nz = 100
start_epochs = args.start_epochs
end_epochs = args.end_epochs
step = args.step
threshould = args.threshold  # Consider adjusting this based on training output distribution

print(f"Using device: {device}")
class_number = f'closed{num_classes}'
# Create unique directory name based on hyperparameters
params_dir = f'exp{exp}_size{size}_numd{num_classes}_seed{seed}_lr{lr}_ngf{ngf}_beta1{beta1}_b{batch_size}'
ablation_dir = os.path.join(f'./experiments_results', class_number, params_dir)

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

# Print which phases will run
print(f"Running phases: Train={args.run_train}, Validation={args.run_val}, Test={args.run_test}")


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


#### Training phase ####
if args.run_train:
    # configure the enrolled device dataset and dataloader
    train_closedset = PUFdataset(f'./SRAM_dataset/size{size}device{num_classes}/closed/train', transform = other_transform )
    train_closed_loader = DataLoader(train_closedset, batch_size = batch_size, shuffle=True, num_workers=4)
    modified_resnet.eval()

    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator(nc=nc, ndf=ndf).to(device)

    netD.apply(weights_init)
    netG.apply(weights_init)

    criterion = nn.BCELoss()

    # # Setup Adam optimizers for both G and D
    optimizerD = optim.AdamW(netD.parameters(), lr=lr/2.5, betas=(beta1, 0.999), weight_decay=1e-2)
    optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-3)

    # Training Loop
    # Lists to keep track of progress
    G_losses = []
    D_overall_losses = []
    D_fake_losses = []
    D_real_losses = []
    D_G_z1_list = []
    D_G_z2_list = []
    D_x_list = []

    D_x_median_list = []
    D_x_variance_list = []  

    iters = 0

    # Define the path of the saved directory
    save_results_dir = os.path.join(ablation_dir, 'train')
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Loss save path: {save_results_dir}")

    weight_save_dir = os.path.join(ablation_dir, 'gan_weights')
    os.makedirs(weight_save_dir, exist_ok=True)
    print(f"Weight save path: {weight_save_dir}")

    real_label = 0.95  # Soft label for real (closed-set) samples
    fake_label = 0.05  # Soft label for fake (generated) samples

    total_batches = len(train_closed_loader)

    # Training OpenGAN
    for epoch in range(1, num_epochs + 1):
        if (epoch%5==0):
            print(f"Epoch [{epoch}/{num_epochs}]")
        for i, ((inputs, _)) in enumerate(train_closed_loader):
            with torch.no_grad():
                inputs = inputs.to(device)
                r_size = inputs.size(0)
                _, real_features = modified_resnet(inputs)
                
            # (1) update D: Train with all-real closed batch for D
            netD.zero_grad()            
            real_features = real_features.to(device)
            
            label_real = torch.full((r_size,), real_label, dtype=torch.float, device=device)
            disc_output = netD(real_features)  # Shape: (batch, 1, 1, 1)
            # Ensure correct shape for loss computation
            if disc_output.dim() > 1:
                logits = disc_output.view(disc_output.size(0), -1).squeeze(-1)  # Flatten to (batch,)
            else:
                logits = disc_output.squeeze()
            output = logits 
            
            errD_real = lamba0 * criterion(logits, label_real)
            D_real_losses.append(errD_real.item())  # List for D_real_losses
            errD_real.backward()
            
            # calculate median, variance, mean
            D_x_median = output.median().item()
            D_x_variance = output.var().item()  
            D_x = output.mean().item()
            
            ## Train with all-fake batch for D
            noise = torch.randn(int(r_size), nz, 1, 1, device=device)
            fake_features = netG(noise)
            label_fake = torch.full((int(r_size),), fake_label, dtype=torch.float, device=device)
            disc_output = netD(fake_features.detach())  # Shape: (batch, 1, 1, 1)
            # Ensure correct shape for loss computation
            if disc_output.dim() > 1:
                logits = disc_output.view(disc_output.size(0), -1).squeeze(-1)  # Flatten to (batch,)
            else:
                logits = disc_output.squeeze()
            
            errD_fake = lamba1 * criterion(logits, label_fake)
            D_fake_losses.append(errD_fake.item()) # List for D_fake_losses

            output = logits
            errD_fake.backward()    
            optimizerD.step()   # Update D         
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake 
                 
            # (2) update G, owing to adversarial training, so we set the label Real for G generated features.
            netG.zero_grad()
            label_G = torch.full((int(r_size),), real_label, dtype=torch.float, device=device) 
            disc_output = netD(fake_features)  # Shape: (batch, 1, 1, 1)
            # Ensure correct shape for loss computation
            if disc_output.dim() > 1:
                logits = disc_output.view(disc_output.size(0), -1).squeeze(-1)  # Flatten to (batch,)
            else:
                logits = disc_output.squeeze()
                
            # output = torch.sigmoid(logits)
            output = logits
            
            errG = criterion(logits, label_G)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()        
            
            G_losses.append(errG.item()) # List for G_losses
            D_overall_losses.append(errD.item())  # List for D_overall_losses
            iters += 1
            D_G_z1_list.append(D_G_z1)
            D_G_z2_list.append(D_G_z2)
            D_x_list.append(D_x)
            D_x_median_list.append(D_x_median)
            D_x_variance_list.append(D_x_variance)  
        
        cur_model_wts = copy.deepcopy(netG.state_dict())
        path_to_save_paramOnly = os.path.join(weight_save_dir, 'epoch-{}.GNet'.format(epoch))
        torch.save(cur_model_wts, path_to_save_paramOnly)
        
        cur_model_wts = copy.deepcopy(netD.state_dict())
        path_to_save_paramOnly = os.path.join(weight_save_dir, 'epoch-{}.DNet'.format(epoch))
        torch.save(cur_model_wts, path_to_save_paramOnly)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_fake_losses,label="D_f")
    plt.plot(D_real_losses,label="D_r")
    plt.plot(D_overall_losses,label="D_all")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    filename1 = 'each part loss for openGAN.png'
    file_path1 = os.path.join(save_results_dir, filename1)
    plt.savefig(file_path1, bbox_inches='tight', transparent=False)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("D(X~) during training epochs")
    plt.plot(D_G_z1_list,label="D_G_z1")
    plt.plot(D_G_z2_list,label="D_G_z2")
    plt.plot(D_x_list,label="D_x")
    plt.plot(D_x_median_list, label="D_x_median")
    plt.plot(D_x_variance_list, label="D_x_variance")
    plt.xlabel("iterations")
    plt.ylabel("Averge outputs for different categeories of examples")
    plt.legend()
    filename3 = 'Averge outputs vs iters.png'
    file_path1 = os.path.join(save_results_dir, filename3)
    plt.savefig(file_path1, bbox_inches='tight', transparent=False)
    plt.show()


    plt.figure(figsize=(10,5))
    plt.title("Average Generator and Discriminator Loss per Epoch")
    plt.plot(D_overall_losses, label="D_all")
    plt.plot(G_losses, label="G_loss")
    plt.xlabel("iterations")
    plt.ylabel("Overall trend of G & D Loss")
    plt.legend()
    filename2 = 'overall loss for OGAN.png'
    file_path2 = os.path.join(save_results_dir, filename2)
    plt.savefig(file_path2, bbox_inches='tight', transparent=False)
    plt.show()


#### Validation phase ####
if args.run_val:

    weight_save_dir = os.path.join(ablation_dir, 'gan_weights')
    os.makedirs(weight_save_dir, exist_ok=True)
    print(f"OpenGAN Weight save path: {weight_save_dir}")

    save_results_dir = os.path.join(ablation_dir, 'validate')
    os.makedirs(save_results_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(seed)  #99  lr = 0.0005
    random.seed(seed)
    np.random.seed(seed)

    other_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    # Configure dataset and dataloader
    val_closedset = PUFdataset(f'./SRAM_split_dataset/size{size}device{num_classes}/closed/val',
                               transform = other_transform)
    val_openset = PUFdataset(f"./SRAM_split_dataset/size{size}device{num_classes}/val_out", 
                             transform = other_transform, target_transform=ToUnknown())
    validation_loader = DataLoader(ConcatDataset([val_closedset, val_openset]), 
                                   batch_size=batch_size, shuffle=False, num_workers=4)

    if closed_model == "res34":
        # Load the well-trained Modified_ResNet34 model
        model_resnet34 = resnet34(weights=None)
        modified_resnet = ModifiedResNet34(model=model_resnet34, num_classes=num_classes)
    elif closed_model == "res18":
        model_resnet18 = resnet18(weights=None)
        modified_resnet = ModifiedResNet18(model=model_resnet18, num_classes=num_classes)
    else:
        raise ValueError("Invalid closed_model. Choose either 'res34' or 'res18'.")

    modified_resnet.load_state_dict(torch.load(closed_weight_path, weights_only=True))
    # 将modified_resnet设置为评估模式
    modified_resnet.to(device)
    modified_resnet.eval()

    netD = Discriminator(nc=nc, ndf=ndf).to(device)

    Accuracy_list = []
    correct_rejection_list = []
    correct_enrolled_list = []

    # Open file to save validation results
    val_results_file = os.path.join(save_results_dir, 'validate.txt')
    val_file = open(val_results_file, 'w')
    val_file.write("Validation Results\n")
    val_file.write("="*80 + "\n")

    for epoch in range(start_epochs, end_epochs + 1, step):
        weights_path = os.path.join(weight_save_dir, f"epoch-{epoch}.DNet")
        if not os.path.exists(weights_path):
            print(f"Weights file for epoch {epoch} not found, skipping.")
            continue
        
        netD.load_state_dict(torch.load(weights_path, map_location=device))
        netD.eval()

        correct = 0
        total = 0
        
        # Initialize the counter
        correct_rejection_total = 0
        correct_enrolled_total = 0
        total_open_samples = 0
        total_enrolled_samples = 0
        
        # Confusion-matrix counters for F1 scores (reset every epoch)
        tp_open = 0
        fp_open = 0
        fn_open = 0
        
        tp_closed = 0
        fp_closed = 0
        fn_closed = 0
        
        # Diagnostic counters
        closed_disc_outputs = []
        open_disc_outputs = []
        closed_classifier_correct = 0
        closed_classifier_total = 0
        
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                
                closed_result, features = modified_resnet(data)
                _, closed_predicted = torch.max(closed_result.data, 1)
                
                # Get discriminator output and ensure correct shape
                disc_output = netD(features)  # Shape: (batch, 1, 1, 1) or (batch, 1)
                # Handle different possible shapes
                if disc_output.dim() > 1:
                    disc_output = disc_output.view(disc_output.size(0), -1).squeeze(-1)  # Flatten to (batch,)
                else:
                    disc_output = disc_output.squeeze()
                
                binary_result = disc_output  # Shape: (batch,)
                
                binary_predictions = (binary_result > threshould)
                
                # Masks for enrolled and outlier samples respectively
                closed_mask = (target != -1)
                open_mask = (target == -1)            

                # Calculate the correctly authenticated enrolled samples
                correct_enrolled = ((binary_predictions[closed_mask] == True) & 
                                (closed_predicted[closed_mask] == target[closed_mask])).sum().item()
                
                # Calculate the successfully rejected outlier samples (rejected by the discriminator)
                correct_rejection = (binary_predictions[open_mask] == False).sum().item()

                total_enrolled_samples += closed_mask.sum().item()
                total_open_samples += open_mask.sum().item()
                            
                # Accumulate the correct number and the total
                correct_enrolled_total += correct_enrolled
                correct_rejection_total += correct_rejection
                
                correct += correct_enrolled + correct_rejection
                total += target.size(0)    
                
        correct_enrolled_rate = correct_enrolled_total / total_enrolled_samples if total_enrolled_samples > 0 else 0
        correct_rejection_rate = correct_rejection_total / total_open_samples if total_open_samples > 0 else 0    

        correct_rejection_list.append(correct_rejection_rate* 100)
        correct_enrolled_list.append(correct_enrolled_rate* 100)
        
        result_str = f"Val epoch {epoch}: FAR={(1-correct_rejection_rate)* 100:.2f}%, FRR={(1-correct_enrolled_rate)* 100:.2f}%"
        print(result_str)
        val_file.write(result_str + "\n")
        
    val_file.close()
    print(f"Validation results saved to: {val_results_file}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(range(start_epochs, end_epochs + 1, step), Accuracy_list,  color='#E07B54', linewidth=2.5, marker='o', markersize=5, label='Overall Validation Accuracy')
    ax.plot(range(start_epochs, end_epochs + 1, step), correct_rejection_list, color="#E1C855", linewidth=2.5, marker='s', markersize=5, label='Open Validation Accuracy')
    ax.plot(range(start_epochs, end_epochs + 1, step), correct_enrolled_list,  color='#51B1B7', linewidth=2.5, marker='^', markersize=5, label='Closed Validation Accuracy')


    ax.set_title("Open-set Validation Accuracies vs Epoch", fontsize=20, color="#0A0A0A")
    ax.set_xlabel("Epoch", fontsize=20, color="#0A0A0A")
    ax.set_ylabel("Accuracy (%)", fontsize=20, color="#0A0A0A")
    ax.grid(True)


    ax.set_xticks(range(start_epochs, end_epochs + 1, step))
    ax.set_xticklabels([str(x) for x in range(start_epochs, end_epochs + 1, step)], rotation=45)

    ax.legend(fontsize=18) 

    plt.tight_layout()

    print(f"validation result save path: {save_results_dir}")
    # save the validation results
    plt.savefig(os.path.join(save_results_dir, f'exp{exp}step{step}val{start_epochs}-{end_epochs}for Vth-{threshould}.pdf'), dpi=300, bbox_inches='tight') # 使用高分辨率保存


#### Test phase ####
if args.run_test:

    weight_save_dir = os.path.join(ablation_dir, 'gan_weights')
    os.makedirs(weight_save_dir, exist_ok=True)
    print(f"OpenGAN Weight save path: {weight_save_dir}")

    save_results_dir = os.path.join(ablation_dir,'test/')
    os.makedirs(save_results_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(seed)  #99  lr = 0.0005
    random.seed(seed)
    np.random.seed(seed)

    # Configure dataset and dataloader
    test_closedset = PUFdataset(f'./SRAM_split_dataset/size{size}device{num_classes}/closed/test', 
                                transform = other_transform)
    test_openset = PUFdataset(f"./SRAM_split_dataset/size{size}device{num_classes}/test_out", 
                              transform = other_transform, target_transform=ToUnknown())
    test_loader = DataLoader(ConcatDataset([test_closedset, test_openset]), batch_size=batch_size, 
                             shuffle=False, num_workers=4)

    modified_resnet.eval()
    netD = Discriminator(nc=nc, ndf=ndf).to(device)

    real_label = 1
    fake_label = 0

    Accuracy_list = []
    correct_rejection_list = []
    correct_enrolled_list = []

    # Open file to save test results
    test_results_file = os.path.join(save_results_dir, 'test.txt')
    test_file = open(test_results_file, 'w')
    test_file.write("Test Results\n")
    test_file.write("="*80 + "\n")

    # Traverse the weight files of each epoch
    for epoch in range(start_epochs, end_epochs + 1, step):
        weights_path = os.path.join(weight_save_dir, f"epoch-{epoch}.DNet")
        if not os.path.exists(weights_path):
            print(f"Weights file for epoch {epoch} not found, skipping.")
            continue
    
        netD.load_state_dict(torch.load(weights_path, map_location=device))
        netD.eval()

        correct = 0
        total = 0
        
        correct_rejection_total = 0
        correct_enrolled_total = 0
        total_open_samples = 0
        total_enrolled_samples = 0  
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                closed_result, features = modified_resnet(data)
                _, closed_predicted = torch.max(closed_result.data, 1)
                
                # Get discriminator output and ensure correct shape (consistent with validation)
                disc_output = netD(features)
                if disc_output.dim() > 1:
                    binary_result = disc_output.view(disc_output.size(0), -1).squeeze(-1)
                else:
                    binary_result = disc_output.squeeze()
                
                binary_predictions = (binary_result > threshould)
                
                # Masks for enrolled and outlier samples respectively
                closed_mask = (target != -1)
                open_mask = (target == -1)            

                correct_enrolled = ((binary_predictions[closed_mask] == True) & 
                                (closed_predicted[closed_mask] == target[closed_mask])).sum().item()
                
                correct_rejection = (binary_predictions[open_mask] == False).sum().item()

                total_enrolled_samples += closed_mask.sum().item()
                total_open_samples += open_mask.sum().item()

                correct_enrolled_total += correct_enrolled
                correct_rejection_total += correct_rejection
                
                correct += correct_enrolled + correct_rejection
                total += target.size(0)    
                        
        correct_enrolled_rate = correct_enrolled_total / total_enrolled_samples 
        correct_rejection_rate = correct_rejection_total / total_open_samples

        correct_rejection_list.append(correct_rejection_rate* 100)
        correct_enrolled_list.append(correct_enrolled_rate* 100)
        
        result_str = f"Test epoch {epoch}: FAR={(1-correct_rejection_rate)* 100:.2f}%, FRR={(1-correct_enrolled_rate)* 100:.2f}%"
        print(result_str)
        test_file.write(result_str + "\n")

    # Close test results file
    test_file.close()
    print(f"Test results saved to: {test_results_file}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8)) # 更改为单个子图

    ax.plot(range(start_epochs, end_epochs + 1, step), Accuracy_list,  color='#E07B54', linewidth=2.5, marker='o', markersize=5, label='Overall test Accuracy')
    ax.plot(range(start_epochs, end_epochs + 1, step), correct_rejection_list, color="#E1C855", linewidth=2.5, marker='s', markersize=5, label='Open test Accuracy')
    ax.plot(range(start_epochs, end_epochs + 1, step), correct_enrolled_list,  color='#51B1B7', linewidth=2.5, marker='^', markersize=5, label='Closed test Accuracy')

    ax.set_title("Open-set test Accuracies vs Epoch", fontsize=20, color="#0A0A0A")
    ax.set_xlabel("Epoch", fontsize=20, color="#0A0A0A")
    ax.set_ylabel("Accuracy (%)", fontsize=20, color="#0A0A0A")
    ax.grid(True)

    ax.set_xticks(range(start_epochs, end_epochs + 1, step))
    ax.set_xticklabels([str(x) for x in range(start_epochs, end_epochs + 1, step)], rotation=45)

    ax.legend(fontsize=18) 

    plt.tight_layout()

    print(f"Test result save path: {save_results_dir}")
    plt.savefig(os.path.join(save_results_dir, f'exp{exp}step{step}test{start_epochs}to{end_epochs}.pdf'), dpi=300, bbox_inches='tight') # 使用高分辨率保存