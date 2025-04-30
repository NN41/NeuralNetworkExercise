# %% Set up

import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# X_np, y_np = make_moons(n_samples=500, noise=0.15, random_state=0)
X_np, y_np = make_blobs(n_samples=500, centers=2, n_features=2, random_state=0, cluster_std=0.5)
plt.scatter(X_np[:,0], X_np[:,1], c=y_np)

# %% Exercise 1

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, output_size=1, activation=None):
        super().__init__()

        self.activation = activation
        self.output_size = output_size

        # build the layers dynamically
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        if activation is not None:
            layers.append(activation) # must be a Pytorch module (nn.Module)
        layers.append(nn.Linear(hidden_size, output_size))

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.stack(x)
        return logits


def plot_predictions(model, choose_class=True, add_to_title=""):

    GRID_RESOLUTION = 0.02
    x1_min, x1_max = X_np[:,0].min(), X_np[:,0].max()
    x2_min, x2_max = X_np[:,1].min(), X_np[:,1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, GRID_RESOLUTION), np.arange(x2_min, x2_max, GRID_RESOLUTION))

    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    if model.output_size == 1:
        grid_pred_class = nn.Sigmoid()(model(grid_tensor))
        if choose_class:
            grid_pred_class = (grid_pred_class > 0.5) * 1
    grid_pred_class = grid_pred_class.detach().cpu().numpy()
    grid_pred_class = grid_pred_class.reshape(xx1.shape)

    plt.figure(figsize=(8,8))
    plt.contourf(xx1, xx2, grid_pred_class, cmap=plt.cm.coolwarm, alpha=0.8, vmin=0, vmax=1)
    plt.scatter(X_np[:,0], X_np[:,1], c=y_np, cmap=plt.cm.coolwarm, edgecolors='k', s=30)
    plt.title("Predicted classes and decision boundary" + add_to_title)


def print_model_info(model):
    print(f"Model structure:\n{model}\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | \n\tSize: {param.shape} | \n\tWeights for first unit in next layer : {param[:1]} \n")

# X = torch.tensor(X_np, dtype=torch.float32, device=device)
# y = torch.tensor(y_np, dtype=torch.float32, device=device)

model = SimpleMLP().to(device) # moves the model to the GPU
print_model_info(model)
plot_predictions(model, choose_class=True)
plot_predictions(model, choose_class=False)

# %% Exercise 2
# We are doing a binary classification using a single output variable. It is useful to estimate
# not the output probabilities sigma(z), but rather the logits (log-probabilities) z, such that
# we can use the nn.BCEWithLogitsLoss loss function for increased numerical stability.
#
# Note how no activation function always gives linear decision boundary, relu gives piecewise-linear,
# and tanh and sigmoid give the same smooth boundary.

HIDDEN_SIZE = 4

model_no_activation = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=None).to(device)
model_relu = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.ReLU()).to(device)
model_sigmoid = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Sigmoid()).to(device)
model_tanh = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Tanh()).to(device)
models = [model_no_activation, model_relu, model_sigmoid, model_tanh]

for m in models:
    print_model_info(model=m)
    plot_predictions(model=m, choose_class=False, add_to_title=f": activation function {m.activation}")
    plot_predictions(model=m, choose_class=True, add_to_title=f": activation function {m.activation}")

# %% Exercise 3 & 4
# We use a single output variable for the binary classification, so we use BCE as a loss function

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

BATCH_SIZE = 64
TEST_SPLIT_FRACTION = 0.2
model = model_no_activation

# X_np, y_np = make_moons(n_samples=500, noise=0.15, random_state=0)
X_np, y_np = make_blobs(n_samples=500, centers=2, n_features=2, random_state=0, cluster_std=0.5)
plt.scatter(X_np[:,0], X_np[:,1], c=y_np)
plt.show()

# create train and test datasets
y_np = y_np.reshape((len(y_np),1))
X = torch.tensor(X_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np, dtype=torch.float32, device=device)
full_dataset = TensorDataset(X, y)
train_dataset, test_dataset = random_split(full_dataset,
                                     [1-TEST_SPLIT_FRACTION, TEST_SPLIT_FRACTION],
                                     torch.Generator().manual_seed(42))

# create dataloader (for batching)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

print(f"Full dataset:\n\tExamples: {len(full_dataset)}\n\tShape of X: {X.shape}\n\tShape of y: {y.shape}")
print(f"Train/test dataloader:\n\tTrain data batches: {len(train_dataloader)}\n\tTest data batches: {len(test_dataloader)}")
print(f"\n\tTrain batches:")
for batch, (Xt, yt) in enumerate(train_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")
print(f"\n\tTest batches:")
for batch, (Xt, yt) in enumerate(test_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")


# %%



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward() # uses back-propagation to compute the gradients of the loss w.r.t. the model weights
        optimizer.step() # updates the model weights based on the gradients
        optimizer.zero_grad() # gradients are accumulated by default, this prepares them for the next minibatch

        loss_value = loss.item()
        current = (batch + 1) * len(X)
        print(f"loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if y.shape[1] == 1: # for binary classification
                correct += ((pred > 0) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")



# %%

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model = SimpleMLP(hidden_size=N_hidden, activation=None).to(device)

EPOCHS = 5
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")