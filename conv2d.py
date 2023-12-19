import torch
import numpy as np
from torchsummary import summary


conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=2)

model = torch.nn.Sequential(conv1).cuda()
input = torch.randn(3, 224, 224).cuda()

output = conv1(input)
print(summary(model, (3, 224, 224), batch_size=1))

print(model)
print(output.shape)

