import torch
import torch.utils.data as Data

BATCH_SIZE = 5 
# 如果 BATCH_SIZE = 8 --> 數據會變 8, 2 這樣的大小
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)     # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)     # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y) # 將資料丟入 TensorDataset
loader = Data.DataLoader(
    dataset = torch_dataset,     # torch TensorDataset format
    batch_size = BATCH_SIZE,     # mini batch size
    shuffle = True,              # random shuffle for training
    num_workers = 2
)

def show_batch():
    for epoch in range(3):   # 這裡其實就是 epochs 的次數
        # 一個 epoch 代表 --> 會依序訓練每個 batch_size 的資料，並訓練完一個 batch_size 的資料就更新一次
        for step, (batch_x, batch_y) in enumerate(loader): # enumerate 是為每一次的數據加上一個標號 step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
            
if __name__ == '__main__':
    show_batch()