import torch
from torch.autograd import Variable

## variable 可以做反向傳播，而 tensor 型態沒辦法

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad = True) # requires_grad 打開才能計算梯度

print(tensor)   # [torch.FloatTensor of size 2x2]
print(variable) # [torch.FloatTensor of size 2x2]

t_out = torch.mean(tensor * tensor)     # x^2
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out)

v_out.backward()  # backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print(variable.grad)
'''
 0.5000  1.0000
 1.5000  2.0000
'''

print(variable)     # variable 的形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # 要這樣才能得到 tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # 要進入到 tensor 形式才能轉換到 numpy
"""
[[ 1.  2.]
 [ 3.  4.]]
"""