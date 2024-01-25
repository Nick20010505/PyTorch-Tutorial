import torch
import torch.nn.functional as F

#方法一 --> 用 class 的方式搭建
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_feature, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x 
    
net = Net(1, 10, 1)

# 方法二 -->快速搭建神經網路的方式
net2 = torch.nn.Sequential( # 用 Sequential 的方式快速搭建
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(net1)     # net1 architecture
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

print(net2)     # net2 architecture
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""