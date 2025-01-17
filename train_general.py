import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

MODEL = "GENERAL"

# Load data
def onehot(data):
    action_map = {"NORTH": 0, "SOUTH": 1, "EAST": 2, "WEST": 3,
                  "NORTHEAST": 4, "NORTHWEST": 5, "SOUTHEAST": 6,
                  "SOUTHWEST": 7, "STAY": 8}
    ret = []
    for d in data:
        z = [0 for _ in range(9)]
        z[action_map[d]] = 1
        ret.append(z)
    return ret

dataset = None
y = None


if MODEL == "LEARNED":
    with open("data/grid5.file") as f:
        data = f.read()
        dataset = json.loads(data)
    y = dataset["policy"]
    y = onehot(y)
    y = np.array(y, dtype=np.int32)
    X = dataset["𝒮"]
    X = np.array(X, dtype=np.float32)
    print(f"Size of X: {X.shape}")
    print(X[:10])
    X.resize(X.shape[0], 4)
    print("Reshaped")
    print(X[:10])
elif MODEL == "GENERAL":
    X = []
    y = []
    files = os.listdir("data/")
    for fno, fname in enumerate(files):
        with open(f"data/{fname}") as f:
            data = f.read()
            dataset = json.loads(data)
        y_i = dataset["policy"]
        y_i = onehot(y_i)
        y_i = np.array(y_i, dtype=np.int32)
        X_i = dataset["𝒮"]
        X_i = np.array(X_i, dtype=np.int32)
        blocked_cells = []
        for j in range(11):
            for i in range(11):
                if not dataset["open"][j][i]:
                    blocked_cells.extend([i, j])
        blocked_append = [blocked_cells.copy() for _ in range(X_i.shape[0])]
        blocked_append = np.array(blocked_append, dtype=np.int32)
        X_i.resize(X_i.shape[0], 4)
        X_i = np.concatenate([X_i, blocked_append], axis=1)
        if fno == 0:
            X = X_i
            y = y_i
        else:
            print(f"Turn: {fno}")
            X = np.concatenate([X, X_i])
            y = np.concatenate([y, y_i])
    print(f"Size of X: {X.shape}")
    print(f"Size of y: {y.shape}")
# ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1)
X = torch.tensor(X_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
print(f"X_train size: {X_train.shape}")
print(f"X_test size: {X_test.shape}")
model = None
if MODEL == "LEARNED":
    model = nn.Sequential(
        nn.Linear(4, 50),
        nn.ReLU(),
        nn.Linear(50, 40),
        nn.ReLU(),
        nn.Linear(40, 30),
        nn.ReLU(),
        nn.Linear(30, 9),
        nn.Sigmoid())
elif MODEL == "GENERAL":
    model = nn.Sequential(
        nn.Linear(32, 70),
        nn.LeakyReLU(),
        nn.Linear(70, 80),
        nn.LeakyReLU(),
        nn.Linear(80, 40),
        nn.LeakyReLU(),
        nn.Linear(40, 30),
        nn.LeakyReLU(),
        nn.Linear(30, 9),
        nn.Softmax(dim=1))
else:
    exit(-1)

    

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 1000
batch_size = 8192
 
for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        Xb = X[i:i+batch_size]
        y_hat = model(Xb)
        yb = y[i:i+batch_size]
        loss = loss_fn(y_hat, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_val_hat = model(X_val)
    loss_val = loss_fn(y_val_hat, y_val)
    print(f'Loss: {loss}, Validation Loss: {loss_val}')


# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_hat = model(X_test)
 
accuracy = (y_hat.round() == y_test).float().mean()
print(f"Accuracy {accuracy}")
