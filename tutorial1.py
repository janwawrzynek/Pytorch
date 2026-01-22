import torch
import numpy as np


x_data =  torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
#Now parameter tensor
w = torch.tensor([[1.0],[2.0],[3.0]], requires_grad=True)  # shape (3, 1)


a = torch.tensor(2.0, requires_grad=True)  # scalar parameter
b = torch.tensor(3.0, requires_grad=True)  # scalar parameter
x = torch.tensor(4.0, requires_grad=True)  # input scalar
y = a + b
z = x * y
print(f'Result z: {z}')  # Should print tensor(20., grad_fn=<MulBackward0>)
print(f"grad_fn of z: {z.grad_fn}")  # Should print <MulBackward0 object at ...>

# important
# torch.mean computes the mean of all elements in the input tensor.
# torch.argmax returns the indices of the maximum values along a specified dimension.
#torch.gather(input, dim, index) gathers values along an axis specified by dim.
# gather is highly optimised compared to advanced indexing/ for loops
data = torch.tensor([[10,11,12,13],
                    [20,21,22,23],
                    [30,31,32,33]
                    ])
indices_to_select = torch.tensor([[2], [0], [3]])

selected_value = torch.gather(data, dim = 1, index = indices_to_select)
print(f"Selected values using torch.gather:\n{selected_value}")
