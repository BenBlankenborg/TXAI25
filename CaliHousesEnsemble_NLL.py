import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = fetch_california_housing()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

class HousingNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(HousingNN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
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

num_models = 5
ensemble_models = []
batch_size = 32
epochs = 20

for n in range(num_models):
    model = HousingNN(dropout_rate=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

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

        print(f'Model {n + 1}, Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}', flush=True)

    ensemble_models.append(model)

def negative_log_likelihood(y_true, y_pred_mean, y_pred_std):
    nll = 0.5 * np.mean(((y_true - y_pred_mean) ** 2) / (y_pred_std ** 2) + np.log(y_pred_std ** 2) + np.log(2 * np.pi))
    return nll

all_outputs = []
for model in ensemble_models:
    model.eval()
    with torch.no_grad():
        all_outputs.append(model(X_test_tensor).cpu().numpy())
all_outputs = np.array(all_outputs)
print(all_outputs.shape)
if all_outputs.ndim == 3:
    all_outputs = all_outputs.squeeze(-1)
predictions_list = all_outputs.T.tolist() 
y_pred_mean = np.mean(all_outputs, axis=0).flatten()
y_pred_std = np.std(all_outputs, axis=0).flatten()

y_true = y_test_tensor.cpu().numpy().flatten()
nll = negative_log_likelihood(y_true, y_pred_mean, y_pred_std)
mae = np.mean(np.abs(y_true - y_pred_mean))

predictions_list = [outputs.flatten().tolist() for outputs in all_outputs.transpose(1, 0)]
ground_truths = y_true.tolist()

with open("ensemble_predictions_run_9.pkl", "wb") as f:
    pickle.dump((predictions_list, ground_truths), f)

print(f"Over 50 test samples\nNLL: {nll:.4f}\nMAE: {mae:.4f}\nAverage std: {np.mean(y_pred_std):.4f}")
