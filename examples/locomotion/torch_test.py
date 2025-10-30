import torch

# 元のテンソルを作成
x = torch.randn(4, 10, 3)
print("元のテンソルの形状:", x.shape)
print(x)

# 次元0と次元1を入れ替える
y = x.transpose(0, 1)
print("transpose後のテンソルの形状:", y.shape)

print(y)

z = x.permute(1, 0, 2)
print("permute後のテンソルの形状:", y.shape)

print(z)

#torch.stack

a = []

for i in range(4):
    b = torch.randn(10,3)
    a.append(b)

c = torch.stack(a, dim=1)
# d = c.transpose(0,1)
print(c.shape)
print(c)

d = torch.stack(a, dim=0)
e = d.transpose(0,1)
print(e.shape)
print(e)
