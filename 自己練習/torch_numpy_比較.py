import numpy as np
import torch ##torch 的功能就跟 numpy 一樣，不過可以運行於 GPU 上

## convert data type
np_data = np.arange(6).reshape((2, 3)) 
torch_data = torch.from_numpy(np_data) ## convert numpy to tensor
tensor2array = torch_data.numpy() ##變成 numpy_array 的形式

print(
    "\nnumpy array:", np_data,         # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,     #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array # [[0 1 2], [3 4 5]]
)

## abs 用法一樣
data = [-1, -2, 1, -2]
tensor = torch.FloatTensor(data) ## convert to 32-bit floating
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

## sin 用法也相同
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean 用法也相同
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

## matrix multiplication 矩陣相乘用法要注意
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

## 正確用法
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)

## 錯誤用法
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
    '\ntorch: ', tensor.flatten().dot(tensor.flatten())     # 這裡要先攤開成1維，1*1 + 2*2 + 3*3...
)