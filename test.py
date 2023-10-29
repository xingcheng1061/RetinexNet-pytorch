# import torch
# from torch import nn
# from utils import upper_int
# import torch.nn.functional as F
#
# kernel_size = 3
# stride = 2
# device = torch.device('cuda:0')
# padding = upper_int(((stride - 1) * 600 - stride + kernel_size) / 2), upper_int(
#     ((stride - 1) * 400 - stride + kernel_size) / 2)
#
# model = nn.Conv2d(3, 16, kernel_size, stride=stride, padding=padding).to(device)
# data = torch.randn((64, 3, 600, 400)).to(device)
# print(data.shape)
# data = model(data)
# print(data.shape)
#
# data = torch.randn((1, 1, 3, 3))
# print(data)
# data = F.interpolate(data, (3, 3), mode='nearest')
# print(data)
import torch
import torch.nn as nn
from utils import same_padding
from same_conv2d import Conv2d
# 假设输入的形状为 (batch_size, channels, height, width)
input_shape = (1, 3, 128, 128)

kernel_size = 3
stride = 6
# 手动计算适当的填充量
print(same_padding(kernel_size,stride))
# 创建Conv2d层
conv = Conv2d(in_channels=3, out_channels=6, kernel_size=kernel_size, stride=stride, padding=100)

# 使用适当的填充设置
output = conv(torch.randn(input_shape))
print(output.shape)