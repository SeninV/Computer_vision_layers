import torch
import numpy as np
from torchsummary import summary


pointwise_conv = torch.nn.Conv2d(3, 16, (1, 1))

model = torch.nn.Sequential(pointwise_conv).cuda()
input = torch.randn(3, 224, 224).cuda()


output = pointwise_conv(input)

summary(model=model, input_size=(3, 224, 224))
print(output.shape)