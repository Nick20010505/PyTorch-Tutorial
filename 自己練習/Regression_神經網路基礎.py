import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # 增加一個維度！pytorch 中會需要多增加一個維度
print(x[0:5, ]) 
print(x.shape) # torch.Size([100, 1])
y = x.pow(2) + 0.2 * torch.rand(x.size()) # add some noise

class Net(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #繼承 Net 的東西，標準流程一定要加
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer 
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x)) # hidden layer 的 activation function
        x = self.predict(x)        # linear output，因為是 regression 所以不用 activation function
        return x 
    
net = Net(n_feature=1, n_hidden=10, n_output=1) #定義一個 network
print(net) # network architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss() # loss function 使用 MSE

plt.ion() # something about plotting

for t in range(200):
    prediction = net(x)

    loss = loss_func(prediction, y) #預測的值一定要在前面

    optimizer.zero_grad() # 訓練前先將 gradient 清零
    loss.backward()
    optimizer.step()

    if t % 5 == 0: # 這裡的 Data 已經是 Variable 的形式
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()