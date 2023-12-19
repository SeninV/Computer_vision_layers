import torch
import numpy as np
from torchsummary import summary



depthwise_conv = torch.nn.Conv2d(3, 3, (3, 3), padding="same", groups=3)
pointwise_conv = torch.nn.Conv2d(3, 16, (1, 1))



depthwise_separable_conv = torch.nn.Sequential(depthwise_conv, pointwise_conv).cuda()

input = torch.randn(3, 224, 224).cuda()


output = pointwise_conv(input)

summary(model=depthwise_separable_conv, input_size=(3, 224, 224))
print(output.shape)