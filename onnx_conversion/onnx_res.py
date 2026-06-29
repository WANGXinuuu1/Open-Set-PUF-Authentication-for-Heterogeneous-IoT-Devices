import torch
import torch.nn as nn
import os
from torchvision.models import resnet18, resnet34
from model_relevant import ModifiedResNet18, Generator, Discriminator

# Define the path for saving the ONNX file
onnx_path = ""
resnet_weight_path = ''
# input dimension for closed-set classifier following your settings
dummy_input = torch.randn(1, 3, 220, 200)   #(1, 3, W, H)
num_classes = 40
model_resnet18 = resnet18(weights=None)
modified_resnet = ModifiedResNet18(model=model_resnet18, num_classes=num_classes)
modified_resnet.load_state_dict(torch.load(resnet_weight_path, weights_only=True))

modified_resnet.eval()

torch.onnx.export(modified_resnet,
                  dummy_input,
                  onnx_path,
                  export_params=True,
                  opset_version=18,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                 )

