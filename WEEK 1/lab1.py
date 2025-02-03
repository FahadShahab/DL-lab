import torch
import numpy as np

# 1. Illustrating the functions for Reshaping, Viewing, Stacking, Squeezing, and Unsqueezing of tensors
tensor = torch.arange(0, 10).reshape(2, 5)
print(f'original tensor : \n{tensor}')

reshaped_tensor = tensor.view(5, 2)
print(f'Reshaped tensor : \n{reshaped_tensor}')

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
stacked_tensor = torch.stack([tensor1, tensor2], dim=0)
print(f"Stacked Tensor (dim=0):\n{stacked_tensor}")

print(f'Tensor after unsquezing : {tensor1.unsqueeze(0)}')
print(f'Tensor after squeezing : {tensor1.squeeze()}')


# 2. Illustrate the use of torch.permute()
tensor = torch.rand(2, 5)
permuted_tensor = tensor.permute(1, 0)
print(f"Permuted Tensor (shape {tensor.shape} -> {permuted_tensor.shape})")
print(f'Original tensor : \n{tensor}\n Permuted tensor : \n{permuted_tensor}')


# 3. Illustrating indexing in tensors
tensor = torch.rand(3, 4)
print(f'Original tensor : \n{tensor}')
print(f'Element at (2, 3) : {tensor[2, 3]}')


# 4. Convert NumPy arrays to tensors and back to NumPy arrays
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy Array:\n{np_array}\n")

tensor_from_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy array:\n{tensor_from_np}\n")

np_from_tensor = tensor_from_np.numpy()
print(f"Back to NumPy Array:\n{np_from_tensor}\n")


# 5. Create a random tensor with shape (7, 7)
random_tensor_7x7 = torch.rand(size=(7, 7))
print(f'random tensor with (7, 7) : \n{random_tensor_7x7}')



# 6. Matrix multiplication
random_tensor_1x7 = torch.randn(1, 7)
print(f"Tensor (1x7):\n{random_tensor_1x7}\n")

result = torch.matmul(random_tensor_7x7, random_tensor_1x7.t())
print(f"Matrix Multiplication Result:\n{result}\n")



# 7. Create two random tensors of shape (2, 3) and send them to the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_a = torch.randn(2, 3).to(device)
tensor_b = torch.randn(2, 3).to(device)
print(f"Tensor A (GPU):\n{tensor_a}\n")
print(f"Tensor B (GPU):\n{tensor_b}\n")


# 8. Matrix multiplication on the tensors (adjust shapes if needed)
result_gpu = torch.matmul(tensor_a, tensor_b.t())
print(f"Matrix Multiplication Result on GPU:\n{result_gpu}\n")




# 9. Find the maximum and minimum values of the output of step 7
max_val = result_gpu.max()
min_val = result_gpu.min()
print(f'Max value : {max_val}')
print(f'Min value : {min_val}')


# 10. Find the maximum and minimum index values of the output of step 7
max_idx = result_gpu.argmax()
min_idx = result_gpu.argmin()
print(f'Max index : {max_idx}')
print(f'Min index : {min_idx}')


# 11. Create a random tensor with shape (1, 1, 1, 10) and remove 1-dimensional entries
