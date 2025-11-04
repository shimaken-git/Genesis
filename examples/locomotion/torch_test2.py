import torch

x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                 [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5], [10.5, 11.5, 12.5]],
                 ])

print(x)
print(x.shape)

y = torch.tensor([2, 3, 4])
z = torch.tensor([1, 1, 1])
x = x * y + z
print(x)