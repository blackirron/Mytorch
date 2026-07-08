import torch

gamer_data = torch.tensor([
    [450.0, 120.0, 8.5, 1.0],
    [12.0, 5.0, 2.1, 0.0],
    [900.0, 400.0, 9.9, 1.0]
])

print(f"Tensor Data:\n{gamer_data}")
print(f"Shape: {gamer_data.shape}")
print(f"Data Type: {gamer_data.dtype}")


flattened_data = gamer_data.view(12)
print(f"Flattened: {flattened_data}")

tensor_3d = gamer_data.view(1, 3, 4)
print(f"3D: {tensor_3d}")

print("\n THE GPU SHIFT ")
if torch.cuda.is_available():
    gpu_tensor = gamer_data.to('cuda')
    print(f"Successfully moved to: {gpu_tensor.device}")
else:
    print("No GPU detected.")

# Calc gradients
weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
weight_labels = ["Weight_A", "Weight_B", "Weight_C"]
output = (weights * 2).sum()

output.backward()

print(f"Mathematical Output Value: {output.item():}\n")
print(f"{'PARAMETER':<12} | {'CURRENT VALUE':<13} | {'GRADIENT (INFLUENCE)':<20}")
print("-" * 53)

for label, val, grad in zip(weight_labels, weights, weights.grad):
    print(f"{label:<12} | {val.item():<13.1f} | {grad.item():<20.1f}")
