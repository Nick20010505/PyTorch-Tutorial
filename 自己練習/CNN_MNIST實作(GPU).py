import os 
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
'''
註解掉的地方單純為可視化的部分

這邊程式碼為轉到 GPU 上，需要 .to(device) 的地方有 b_x, b_y, test_x, model
且如果需要將在GPU上的數值轉為 numpy，則需先轉回到 cpu --> .cpu()
eg. loss.cpu().item()  pred_y.cpu().numpy() 
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# hyperparameters 
EPOCH = 5
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# mnist digits dataset
if not(os.path.exists('./mnist')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

print(DOWNLOAD_MNIST) 

# preparing training dataset
train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train = True ,                                    # 下載訓練資料
    transform = torchvision.transforms.ToTensor(),    # 將資料型態轉為 tensor
    download = DOWNLOAD_MNIST
)

# plot one example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# 依照 batch size 進行分批
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

# preparing testing dataset
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
'''
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
print(test_x.shape)
'''
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# 多一維因為要有 channel 的維度
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]
test_x = test_x.to(device)
# build CNN network
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (1, 28, 28)
        self.conv1 = nn.Sequential(     # 這裡只用 nn 而不是 torch.nn 是因為前面命名過了
            nn.Conv2d(
                in_channels = 1,          # input channel 的數量
                out_channels = 16,      # output 的維度，也等於 filter 的數量
                kernel_size = 5,        # filter 的大小
                stride = 1,             # 步伐的大小
                padding = 2             # padding=(kernel_size-1)/2 if stride=1
            ),                          # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 一個 conv2d 通常都包含這三個
        )                               # output shape (16, 14, 14)
        self.conv2 = nn.Sequential(     # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels = 16,         # input channel 的數量，上一層 output 的大小
                out_channels = 32,       
                kernel_size = 5,         
                stride = 1,              
                padding = 2              
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10) # 輸出是 10 個類別
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)        # (batch, 32, 7, 7) --> (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x                 # return x for visualization
    
cnn = CNN()
print(cnn)    # network architecture
cnn.to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
'''
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
'''

# training and testing 
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        out = cnn(b_x)[0]            # 因為我們 output 有兩個
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            # 這邊因為 test_output 為 CUDA tensor 形式，所以要先轉回 cpu tensor 才能 .numpy()
            pred_y = torch.max(test_output, 1)[1].cpu().numpy()  # 得到最大值的 index ，也可代表數字
            accuracy = float((pred_y == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().item(), '| test accuracy: %.2f' % accuracy)
            '''
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()
'''
# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')