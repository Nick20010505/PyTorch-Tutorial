from random import shuffle
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# hyperparameters 
EPOCH = 5
BATCH_SIZE = 64
TIME_STEP = 28    # 可以看做是 每次給模型 TIME_STEP 個連續時間序列的數據(image height)
INPUT_SIZE = 28   # rnn input size / image width
LR = 0.01
DOWNLOAD_MNIST = True

# download dataset
train_data = dsets.MNIST(
    root = './mnist/',
    train = True,
    transform = transforms.ToTensor(),  # 將數據轉為 tensor 形式
    download = DOWNLOAD_MNIST   
)

# training data
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

# testing data
test_data = dsets.MNIST(
    root = './mnist/',
    train = False,
    transform = transforms.ToTensor()
)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000]
test_x = test_x.to(device)

# define a RNN network
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,     # 每一列的數量
            hidden_size = 64,            # 可以看做是 output 的數量
            num_layers = 1,              # LSTM 的堆疊層數，默認為1
            batch_first = True           # 輸入格式為(batch, time_step, input_size) --> batch 在第一個
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # r_out 是所有時間步的輸出序列，h_n 和 h_c 分別是最後一個時間步的隱藏狀態和記憶狀態
        rout, (h_n, h_c) = self.rnn(x, None)  # None 代表不指定初始的隱藏狀態，讓模型從零開始學習
        
        out = self.out(rout[:, -1, :])        # 這裡選擇了 r_out 序列的最後一個時間步的輸出    
        return out 
    
rnn = RNN()
print(rnn)
rnn = rnn.to(device)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, TIME_STEP, 28)        # reshape x to (batch, time_step, input_size)
        b_x, b_y = b_x.to(device), b_y.to(device)
        
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = rnn(test_x)    # (samples, time_step, input_size)
            pred_y = torch.max(test_out, 1)[1].cpu().numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().item(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')