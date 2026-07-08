import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

class GamerSkillPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

X_train = torch.randn(100, 3)
Y_train = torch.randn(100, 1)
criterion = nn.MSELoss()

def train_model(learning_rate, epochs=100):
    model = GamerSkillPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(epochs):
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    return loss_history

print("Training with LR = 0.0001 (Too Low)...")
loss_low = train_model(0.0001)

print("Training with LR = 0.01 (Optimal)...")
loss_optimal = train_model(0.01)

print("Training with LR = 1.0 (Too High)...")
loss_high = train_model(1.0)

plt.figure(figsize=(10, 6))

plt.plot(loss_low, label='Too Low (LR=0.0001)', color='blue', linewidth=2)
plt.plot(loss_optimal, label='Optimal (LR=0.01)', color='green', linewidth=2)
plt.plot(loss_high, label='Too High (LR=1.0)', color='red', linewidth=2)

plt.title('How Learning Rate Affects MSE (Loss Curve)')
plt.xlabel('Epochs (Training Steps)')
plt.ylabel('Mean Squared Error (Loss)')
plt.ylim(0, max(loss_low) * 1.5) 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
