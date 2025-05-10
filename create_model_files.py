"""
Create placeholder model files for the trading system.
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Get the absolute path to the models directory
models_dir = os.path.abspath("models")
print(f"Creating model files in: {models_dir}")

# Create a simple PyTorch model for exit_optimization.pt
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_actions=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

# Create exit_optimization model
exit_model = PolicyNetwork()
torch.save(exit_model.state_dict(), os.path.join(models_dir, "exit_optimization.pt"))
print("Created exit_optimization.pt")

# Create pattern_recognition model (similar structure)
class PatternRecognitionCNN(nn.Module):
    def __init__(self, num_classes=9, lookback=20, channels=5):
        super(PatternRecognitionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Calculate the size of the flattened features
        flat_size = (lookback // 8) * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool1d(2)(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool1d(2)(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(2)(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

pattern_model = PatternRecognitionCNN()
torch.save(pattern_model.state_dict(), os.path.join(models_dir, "pattern_recognition.pt"))
print("Created pattern_recognition.pt")

# Create ranking_model.pkl
ranking_model_data = {
    'model': None,
    'scaler': None,
    'feature_names': ['price_to_sma_50', 'volume_change', 'rsi_14', 'macd', 'bollinger_width']
}
with open(os.path.join(models_dir, "ranking_model.pkl"), "wb") as f:
    pickle.dump(ranking_model_data, f)
print("Created ranking_model.pkl")

# Create sentiment model directory and files
sentiment_dir = os.path.join(models_dir, "sentiment_model")
os.makedirs(sentiment_dir, exist_ok=True)
with open(os.path.join(sentiment_dir, "labels.json"), "w") as f:
    f.write('["negative", "neutral", "positive"]')
print("Created sentiment_model directory and files")

print("All model files created successfully")
