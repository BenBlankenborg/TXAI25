import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = fetch_california_housing()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)

# Generate out-of-distribution data
num_ood_samples = 10
num_features = X.shape[1]
ood_data = np.random.normal(loc=0, scale=1, size=(num_ood_samples, num_features)) * 4

# Convert to PyTorch tensors and send to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
ood_tensor = torch.tensor(ood_data, dtype=torch.float32).to(device)

# Define neural network with dropout
class HousingNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(HousingNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize model, loss, optimizer
model = HousingNN(0.1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training loop
epochs = 50
batch_size = 16
n_batches = len(X_train_tensor) // batch_size

for epoch in range(epochs):
    model.train()
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_tensor[start:end]
        y_batch = y_train_tensor[start:end]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}', flush=True)

# Enable dropout layers at inference time
def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

# Plot and save prediction distributions
def plot_and_save_distributions(data, label, num, amount):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=20, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title(f"Distribution {num+1}, Label = {label}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    mean = np.mean(data)
    std = np.std(data)
    plt.subplots_adjust(bottom=0.25)
    txt = f"Over {amount} runs, mean = {mean:.2f}, std = {std:.2f}"
    plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=12)

    file_name = f"UQ_Distribution_CHD_OOD_{num+1}.png"
    plt.savefig(file_name)
    plt.close()

# --- Test Set Evaluation ---
total_nll_test = 0
total_std_test = 0
nlls_test = []
predictions = []
ground_truths = []
amount_of_mc_dropout_forward_passes = 5000

print("\n--- Test Set ---")
for d, (x, label) in enumerate(zip(X_test_tensor, y_test_tensor)):
    distribution = []
    x = x.unsqueeze(0)  # Add batch dimension
    for i in range(amount_of_mc_dropout_forward_passes):
        with torch.no_grad():
            output = model(x)
        distribution.append(output.item())
        print(f"\rProgress: {i+1}/{amount_of_mc_dropout_forward_passes}", end='', flush=True)

    mean = np.mean(distribution)
    std = np.std(distribution)
    std = max(std, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * std**2) + ((label.item() - mean) ** 2) / (2 * std**2)
    total_nll_test += nll
    total_std_test += std
    nlls_test.append(nll)
    predictions.append(np.array(distribution))  
    ground_truths.append(label.item())              
    print(f"\nTest data point {d+1}: NLL = {nll:.4f}")
    plot_and_save_distributions(distribution, label.item(), d, amount_of_mc_dropout_forward_passes)

avg_nll_test = total_nll_test / len(X_test_tensor)
avg_std_test = total_std_test / len(X_test_tensor)
print(f"\nAverage NLL over test set: {avg_nll_test:.4f}")
print(f"\nAverage Standard deviation over test set: {avg_std_test:.4f}")

with open("predictions_run_10.pkl", "wb") as f:
    pickle.dump((predictions, ground_truths), f)

# --- OOD Evaluation ---
total_nll_ood = 0
total_std_ood = 0
nlls_ood = []

print("\n--- OOD Set ---")
for d, x in enumerate(ood_tensor):
    distribution = []
    x = x.unsqueeze(0)  # Add batch dimension
    for i in range(amount_of_mc_dropout_forward_passes):
        with torch.no_grad():
            output = model(x)
        distribution.append(output.item())
        print(f"\rProgress: {i+1}/{amount_of_mc_dropout_forward_passes}", end='', flush=True)

    mean = np.mean(distribution)
    std = np.std(distribution)
    std = max(std, 1e-6)
    # Use label = -1 as dummy for OoD points (not used in NLL formula)
    nll = 0.5 * np.log(2 * np.pi * std**2) + (mean ** 2) / (2 * std**2)
    total_nll_ood += nll
    total_std_ood += std
    nlls_ood.append(nll)
    print(f"\nOOD data point {d+1}: NLL = {nll:.4f}")
    plot_and_save_distributions(distribution, 'OoD', d, amount_of_mc_dropout_forward_passes)

avg_nll_ood = total_nll_ood / len(ood_tensor)
avg_std_ood = total_std_ood / len(ood_tensor)
print(f"\nAverage NLL over OoD set: {avg_nll_ood:.4f}")
print(f"\nAverage Standard deviation over OoD set: {avg_std_ood:.4f}")
