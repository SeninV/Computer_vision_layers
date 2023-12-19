import torch
import numpy as np
from torchsummary import summary


depthwise_conv = torch.nn.Conv2d(3, 3, (3, 3), padding="same", groups=3)

model = torch.nn.Sequential(depthwise_conv).cuda()
input = torch.randn(3, 224, 224).cuda()


output = depthwise_conv(input)

summary(model=model, input_size=(3, 224, 224))
print(output.shape)