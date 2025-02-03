import torch
import torch.nn.functional as F

image = torch.rand(6, 6)
image = image.unsqueeze(dim=0)
image = image.unsqueeze(dim=0)

kernel = torch.ones(3, 3)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)

outimage_stride_1 = F.conv2d(image, kernel, stride=1, padding=0)
print("Outimage with stride=1: ", outimage_stride_1.shape)

outimage_stride_2 = F.conv2d(image, kernel, stride=2, padding=0)
print("Outimage with stride=2: ", outimage_stride_2.shape)

outimage_padding_1 = F.conv2d(image, kernel, stride=1, padding=1)
print("Outimage with padding=1: ", outimage_padding_1.shape)

outimage_padding_2 = F.conv2d(image, kernel, stride=1, padding=2)
print("Outimage with padding=2: ", outimage_padding_2.shape)

num_params = kernel.numel()
print("Total number of parameters in the kernel:", num_params)
