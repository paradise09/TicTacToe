import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters)
        )

        self.activate = nn.ReLU()

    def forward(self, x):
        shortcut = x

        x = self.conv(x)
        x += shortcut
        x = self.activate(x)

        return x
    
'''
Input shape : (3, 3, 2) -> my state (3 x 3) + enemy state (3 x 3)
Output shape : 9 -> available positions for placing pieces (3 x 3)
'''
class DualNetwork(nn.Module):
    def __init__(self, num_residual_block, num_filters):
        super(DualNetwork, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False), # input shape = (3, 3, 2)
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.res_block = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_block)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy = nn.Linear(num_filters, 9) # output shape = 9
        self.value = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        policy = F.softmax(self.policy(x), dim=1)
        value = torch.tanh(self.value(x))
        return policy, value
    
if __name__ == '__main__':
    net = DualNetwork(num_residual_block=16, num_filters=128)
    random_input = torch.randn(3, 3, 2).unsqueeze(0)
    policy, value = net(random_input)
    print(f'{policy}, {value}')
