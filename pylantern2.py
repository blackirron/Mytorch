import torch
import torch.nn as nn

class GamerSkillPredictor(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super(GamerSkillPredictor, self).__init__()

        self.layer1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_units, out_features=output_features)

    def forward(self, x):
        x = self.layer1(x)   # Math: X * W1^T + b1
        x = self.relu(x)     # Math: max(0, x)
        x = self.layer2(x)   # Math: X * W2^T + b2
        return x

model = GamerSkillPredictor(input_features=3, hidden_units=8, output_features=1)
print("Network Architecture:")
print(model)

dummy_gamer_data = torch.randn(5, 3)

predictions = model(dummy_gamer_data)

print("\n--- FORWARD PASS RESULTS ---")
print(f"Input Shape: {dummy_gamer_data.shape} (5 players, 3 features)")
print(f"Output Shape: {predictions.shape} (5 players, 1 prediction each)")
print(f"Raw Predictions:\n{predictions}")
