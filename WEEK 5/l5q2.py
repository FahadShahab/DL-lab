import torch
import torch.nn as nn
import torch.nn.functional as F

image = torch.rand(6, 6)
image = image.unsqueeze(dim=0)
image = image.unsqueeze(dim=0)

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)

outimage_conv2d = conv_layer(image)
print("Output of Conv2d: ", outimage_conv2d.shape)
print("Image : ", outimage_conv2d)



image = torch.rand(6, 6)
image = image.unsqueeze(dim=0)
image = image.unsqueeze(dim=0)  #

kernel = torch.rand(3, 1, 3, 3)

outimage_func_conv2d = F.conv2d(image, kernel, stride=1, padding=0)
print("Output of F.conv2d : ", outimage_func_conv2d.shape)
print("Image : ", outimage_func_conv2d)
