import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)  # Ensure y is a column vector

OOD = True
CLOSE_BY = False
if CLOSE_BY:
    coords = [
        # Oregon (OR)
        [45.5152, -122.6784],  # Portland
        [44.0521, -123.0868],  # Eugene
        [44.9429, -123.0351],  # Salem
        [45.4946, -122.4308],  # Gresham
        [42.3259, -122.8756],  # Medford
        [44.0582, -121.3153],  # Bend
        [45.3007, -122.7730],  # Tigard
        [45.4871, -122.8037],  # Beaverton
        [42.4390, -123.3270],  # Grants Pass
        [45.2160, -123.1970],  # McMinnville
        [45.8491, -122.8183],  # St. Helens
        [42.1882, -120.3450],  # Lakeview
        [44.6341, -124.0534],  # Newport
        [42.1593, -123.6455],  # Cave Junction
        [44.8429, -123.2151],  # Monmouth
        [46.1881, -123.8313],  # Astoria
        [44.2898, -123.0816],  # Junction City

        # Nevada (NV)
        [36.1699, -115.1398],  # Las Vegas
        [39.5296, -119.8138],  # Reno
        [39.1638, -119.7674],  # Carson City
        [36.0111, -114.7378],  # Boulder City
        [40.9720, -117.7357],  # Winnemucca
        [39.2460, -114.8896],  # Ely
        [38.9635, -119.7892],  # Gardnerville
        [36.8055, -114.0672],  # Mesquite
        [39.4650, -118.7755],  # Fallon
        [39.6120, -119.2254],  # Fernley
        [37.9278, -114.0450],  # Pioche
        [39.1483, -119.7589],  # Minden
        [38.9302, -119.7233],  # Topaz Ranch Estates
        [39.5870, -119.8497],  # Spanish Springs
        [38.9773, -117.2315],  # Tonopah
        [41.9783, -117.7353],  # McDermitt

        # Arizona (AZ)
        [33.4484, -112.0740],  # Phoenix
        [32.2226, -110.9747],  # Tucson
        [35.1983, -111.6513],  # Flagstaff
        [33.4152, -111.8315],  # Mesa
        [34.5400, -112.4685],  # Prescott
        [32.6790, -114.5900],  # Yuma
        [35.2481, -113.2220],  # Kingman
        [33.3528, -111.7890],  # Chandler
        [33.5779, -112.1976],  # Glendale
        [33.3076, -111.8449],  # Gilbert
        [34.1469, -114.2892],  # Lake Havasu City
        [33.4636, -112.3452],  # Avondale
        [32.6265, -109.8800],  # Safford
        [35.2601, -111.6937],  # Bellemont
        [31.7054, -110.0616],  # Bisbee
        [32.0350, -109.8187],  # Willcox
        [34.7690, -112.4536],  # Chino Valley
    ]
else:
    coords = [
        [34.5289, 69.1725],   # Kabul, Afghanistan
        [41.3275, 19.8189],   # Tirana, Albania
        [36.7525, 3.0420],    # Algiers, Algeria
        [42.5078, 1.5211],    # Andorra la Vella, Andorra
        [-8.8368, 13.2343],   # Luanda, Angola
        [17.1270, -61.8460],  # Saint John's, Antigua and Barbuda
        [-34.6037, -58.3816], # Buenos Aires, Argentina
        [40.1792, 44.4991],   # Yerevan, Armenia
        [-35.2820, 149.1287], # Canberra, Australia
        [48.2082, 16.3738],   # Vienna, Austria
        [40.4093, 49.8671],   # Baku, Azerbaijan
        [25.2048, 55.2708],   # Abu Dhabi, United Arab Emirates
        [50.8503, 4.3517],    # Brussels, Belgium
        [6.5244, 3.3792],     # Porto-Novo, Benin
        [27.4728, 89.6390],   # Thimphu, Bhutan
        [-15.8267, -47.9218], # Brasília, Brazil
        [42.6977, 23.3219],   # Sofia, Bulgaria
        [12.5657, 104.9910],  # Phnom Penh, Cambodia
        [45.4215, -75.6972],  # Ottawa, Canada
        [9.0579, 7.4951],     # N'Djamena, Chad
        [-33.4489, -70.6693], # Santiago, Chile
        [39.9042, 116.4074],  # Beijing, China
        [4.7110, -74.0721],   # Bogotá, Colombia
        [9.7489, -83.7534],   # San José, Costa Rica
        [45.8150, 15.9819],   # Zagreb, Croatia
        [23.1353, -82.3589],  # Havana, Cuba
        [35.8997, 14.5146],   # Valletta, Malta
        [55.6761, 12.5683],   # Copenhagen, Denmark
        [11.8251, 42.5903],   # Djibouti, Djibouti
        [18.4861, -69.9312],  # Santo Domingo, Dominican Republic
        [-0.2295, -78.5243],  # Quito, Ecuador
        [30.0444, 31.2357],   # Cairo, Egypt
        [13.6929, -89.2182],  # San Salvador, El Salvador
        [59.4370, 24.7536],   # Tallinn, Estonia
        [64.1355, -21.8954],  # Reykjavik, Iceland
        [53.3498, -6.2603],   # Dublin, Ireland
        [41.9028, 12.4964],   # Rome, Italy
        [35.6895, 139.6917],  # Tokyo, Japan
        [31.7683, 35.2137],   # Jerusalem, Israel
        [1.3521, 103.8198],   # Singapore, Singapore
        [37.5665, 126.9780],  # Seoul, South Korea
        [39.9334, 32.8597],   # Ankara, Turkey
        [55.7558, 37.6173],   # Moscow, Russia
        [38.9072, -77.0369],  # Washington, D.C., USA
        [51.5074, -0.1278],   # London, United Kingdom
        [48.8566, 2.3522],    # Paris, France
        [52.5200, 13.4050],   # Berlin, Germany
        [19.4326, -99.1332],  # Mexico City, Mexico
        [-33.9249, 18.4241],  # Cape Town, South Africa
        [28.6139, 77.2090],   # New Delhi, India
    ]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)

if OOD:
    X_inv = scaler.inverse_transform(X_test)
    for i, sample in enumerate(X_inv):
        sample[6] = coords[i][0]
        sample[7] = coords[i][1]
    X_test = scaler.transform(X_inv)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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


# Train multiple models
num_models = 5  # Number of models in the ensemble
ensemble_models = []

# Hyperparameters
batch_size = 32  # Typical batch size
epochs = 20

for n in range(num_models):
    model = HousingNN(dropout_rate=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    # Convert training data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Train model with mini-batches
    n_batches = len(X_train_tensor) // batch_size  # Number of batches per epoch

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

    ensemble_models.append(model)  # Save the trained model

all_outputs = []

for model in ensemble_models:
    model.eval()
    with torch.no_grad():
        all_outputs.append(model(X_test_tensor).numpy())
all_outputs = np.array(all_outputs)
stds = np.std(all_outputs, axis=0)
average_std = np.mean(stds)
if OOD:
    D = 'OoD'
else:
    D = 'ID'
print(f"Average std over 50 {D} data points: {average_std}")