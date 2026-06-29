import os, random
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model_relevant import ModifiedResNet18, ModifiedResNet34
# add PUFdataset_vis to dataset_relevant.py (see PUFdataset_vis.py)
from dataset_relevant import PUFdataset_vis
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# ---- color schemes ----
import matplotlib.cm as cm

# Argument parser
parser = argparse.ArgumentParser(description='UMAP of closed-set PUF features')
parser.add_argument('--exp', type=int, default=205)
parser.add_argument('--size', type=int, default=50)
parser.add_argument('--num_d', type=int, default=45)
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--b', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda:3')
args = parser.parse_args()

size = args.size
num_classes = args.num_d
seed = args.seed
lr = args.lr
ngf = args.ngf
beta1 = args.beta1
batch_size = args.b
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
exp = args.exp
closed_model = 'res18'
print(f"Using device: {device}")
class_number = f'closed{num_classes}'
params_dir = f'exp{exp}results_size{size}_numd{num_classes}_seed{seed}_lr{lr}_ngf{ngf}_beta1{beta1}_b{batch_size}'
ablation_dir = os.path.join(f'./experiments_results', class_number, params_dir)
closed_weight_path = f'./experiments_results/closed45/resnet_weights/best_res18_45puf_size50.pth'

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

if closed_model == "res18":
    modified_resnet = ModifiedResNet18(model=resnet18(weights=None), num_classes=num_classes)
else:
    raise ValueError("Invalid closed_model.")

modified_resnet.load_state_dict(torch.load(closed_weight_path, weights_only=True))
modified_resnet.to(device)
modified_resnet.eval()

save_results_dir = os.path.join(ablation_dir, 'test/')
os.makedirs(save_results_dir, exist_ok=True)

test_closedset = PUFdataset_vis(
    f'./SRAM_dataset/size{size}device{num_classes}/closed/test',
    transform=other_transform)
test_loader = DataLoader(test_closedset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---- extract 512-d features (the same features fed to the classifier head) ----
all_features = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        _, features = modified_resnet(data)
        all_features.append(features.view(features.size(0), -1).cpu().numpy())

feats = np.concatenate(all_features, axis=0)             # (N, 512)
devices = np.array(test_closedset.devices)               # individual PUF per sample
technologies = np.array(test_closedset.technologies)     # 'Arbiter' / 'SRAM'
assert len(devices) == len(feats), f"meta({len(devices)}) != feats({len(feats)})"

print("\n[Visualization] Preparing features for UMAP ...")

# standardize 512-d features
Xs = StandardScaler().fit_transform(feats)

N = Xs.shape[0]
MAX_PTS = 6000
if N > MAX_PTS:
    sel = np.random.RandomState(seed).choice(N, MAX_PTS, replace=False)
    print(f"  Subsampling {MAX_PTS}/{N} points.")
else:
    sel = np.arange(N)
Xs_s = Xs[sel]
dev_s = devices[sel]
tech_s = technologies[sel]

def _safe_sil(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return float('nan')
    return silhouette_score(X, labels)

sil_tech = _safe_sil(Xs_s, tech_s)
sil_device = _safe_sil(Xs_s, dev_s)
if np.isnan(sil_tech) or np.isnan(sil_device):
    verdict = "inconclusive"
elif sil_tech > sil_device:
    verdict = "clusters more by TECHNOLOGY (Arbiter/SRAM)"
else:
    verdict = "clusters more by DEVICE"
sil_line = (f"Silhouette(512-d): technology={sil_tech:.4f}, "
            f"device={sil_device:.4f}  ->  {verdict}")
print("  " + sil_line)

# embeddings (UMAP only)
embeddings = {}
try:
    import umap
except ImportError:
    raise ImportError("umap-learn not installed -> pip install umap-learn")

print("  Running UMAP ...")
reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.6, random_state=99)
embeddings['UMAP'] = reducer.fit_transform(Xs_s)


uniq_dev = sorted(set(dev_s.tolist()))
cmap = plt.get_cmap('turbo')
colors = [cmap(i / max(1, len(uniq_dev) - 1)) for i in range(len(uniq_dev))]

color_rng = np.random.RandomState(42)
color_rng.shuffle(colors)
dev_to_color = {d: colors[i] for i, d in enumerate(uniq_dev)}


def get_marker(tech_name):
    tech_upper = tech_name.upper()
    if 'SRAM' in tech_upper:
        return '^'  # Triangle
    elif 'ARBITER' in tech_upper:
        return 's'  # Square
    return 's'      # Default: Square

for name, emb in embeddings.items():
    fig, ax = plt.subplots(figsize=(12, 9))

    for dev in uniq_dev:
        mask = (dev_s == dev)
        if not np.any(mask): continue
        dev_tech = tech_s[mask][0]
        marker = get_marker(dev_tech)
        color = dev_to_color[dev]

        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[color], marker=marker,
                   s=45, alpha=0.75, edgecolors='white', linewidth=0.5,
                   label=dev)
    ax.set_xticks([])
    ax.set_yticks([])

    tech_handles = [
        Line2D([0], [0],
               marker='^',
               linestyle='None',
               color='w',
               markerfacecolor='gray',
               markeredgecolor='white',
               markeredgewidth=0.5,
               alpha=0.75,
               markersize=10,
               label='SRAM PUF'),
        Line2D([0], [0],
               marker='s',
               linestyle='None',
               color='w',
               markerfacecolor='gray',
               markeredgecolor='white',
               markeredgewidth=0.5,
               alpha=0.75,
               markersize=10,
               label='Arbiter PUF')
    ]

    leg = ax.legend(handles=tech_handles, loc='upper right', title="PUF Type",
                    alignment='left', fontsize=20, title_fontsize=20,
                    frameon=True, framealpha=0.9)

    file_name = f"{name.lower().replace('-', '')}.pdf"
    out_png = os.path.join(save_results_dir, file_name)

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name} figure -> {out_png}")