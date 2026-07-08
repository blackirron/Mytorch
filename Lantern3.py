import torch
import torch.nn as nn

class GamerSkillPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

torch.manual_seed(42)

model = GamerSkillPredictor()

X_train = torch.randn(100, 3)
Y_train = torch.randn(100, 1)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epochs = 100 # no. of times we show the full dataset to the model

print("Starting Training...")
for epoch in range(epochs):

    predictions = model(X_train)

    loss = criterion(predictions, Y_train)
    optimizer.zero_grad() # clear old math before the new step.


    loss.backward()

    optimizer.step() #apply the gradients to adjust the weights slightly

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training Complete! The weights are now optimized.")
