import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
## 現在已捨棄 variable ， 因為 tensor 即可做梯度計算


# define a function that can save model as .pkl
def save():
    # define a network
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()
    
    for t in range(150):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("Net1")
    plt.scatter(x.numpy(), y.numpy()) # x, y 是 tensor，所以可以不用 .data
    plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
    # 有兩種儲存模型的方式
    torch.save(net1, "net.pkl")                      # 儲存整個模型，比較不建議
    torch.save(net1.state_dict(), "net_params.pkl")  # 儲存所有的 parameters
    
# 讀取整個模型
def restore_net():
    net2 = torch.load("net.pkl")
    prediction = net2(x)
    
    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
# 讀取儲存的參數
def restore_params():
    # 因為只保留參數，所以要重新寫一個一樣的模型
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    
    # copy all parameters
    net3.load_state_dict(torch.load("net_params.pkl")) # 用 load_state_dict
    prediction = net3(x)
    
    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()