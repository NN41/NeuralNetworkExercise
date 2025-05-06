# %% Set up

import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device\n")

# %%

NUM_CLASSES = 3
# USE_ONE_HOT = False
# if NUM_CLASSES > 2:
#     USE_ONE_HOT = True
# OUTPUT_SIZE = NUM_CLASSES if NUM_CLASSES > 2 or USE_ONE_HOT else 1

# X_np, y_np = make_moons(n_samples=500, noise=0.15, random_state=0)
X_np, y_np = make_blobs(n_samples=1000, centers=NUM_CLASSES, n_features=2, random_state=0, cluster_std=1.5)
plt.scatter(X_np[:,0], X_np[:,1], c=y_np)

X = torch.tensor(X_np, dtype=torch.float32, device=device)
# y_class_values = torch.tensor(y_np.reshape((len(y_np),1)), device=device)
# y_one_hot = nn.functional.one_hot(y_class_values)
# y = y_one_hot if USE_ONE_HOT else y_class_values
# y = torch.tensor(y_np.reshape)

# torch.tensor(y_np)
y_class_values = torch.tensor(y_np).to(device)
y_one_hot = nn.functional.one_hot(y_class_values)

# %% Exercise 1

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=OUTPUT_SIZE, activation=None):
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
    x1_diff = x1_max - x1_min
    x2_diff = x2_max - x2_min
    x1_min, x1_max = x1_min - x1_diff * 0.05, x1_max + x1_diff * 0.05
    x2_min, x2_max = x2_min - x2_diff * 0.05, x2_max + x2_diff * 0.05
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, GRID_RESOLUTION), 
                           np.arange(x2_min, x2_max, GRID_RESOLUTION))

    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    if model.output_size == 1:
        grid_pred_class = nn.Sigmoid()(model(grid_tensor))
        if choose_class:
            grid_pred_class = (grid_pred_class > 0.5) * 1
    else:
        grid_pred_class = nn.Softmax()(model(grid_tensor)).argmax(dim=1)
    grid_pred_class = grid_pred_class.detach().cpu().numpy()
    grid_pred_class = grid_pred_class.reshape(xx1.shape)

    cmap_to_use = plt.cm.tab10
    vmax_val = max(1, model.output_size-1)

    plt.figure(figsize=(8,8))
    plt.contourf(xx1, xx2, grid_pred_class, cmap=cmap_to_use, alpha=0.8, vmin=0, vmax=vmax_val)
    plt.scatter(X_np[:,0], X_np[:,1], c=y_np, cmap=cmap_to_use, edgecolors='k', s=30)
    plt.title("Predicted classes and decision boundary" + add_to_title)


def print_model_info(model):
    print(f"Model structure:\n{model}\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | \n\tSize: {param.shape} | \n\tWeights for first unit in next layer : {param[:1]} \n")


model = SimpleMLP().to(device) # moves the model to the GPU
print_model_info(model)
plot_predictions(model, choose_class=True) # set choose_class to False for binary classification to see predicted probabilities

# %% Exercise 2
# We are doing a binary classification using a single output variable. It is useful to estimate
# not the output probabilities sigma(z), but rather the logits (log-probabilities) z, such that
# we can use the nn.BCEWithLogitsLoss loss function for increased numerical stability.
#
# Note how no activation function always gives linear decision boundary, relu gives piecewise-linear,
# and tanh and sigmoid give the same smooth boundary.

HIDDEN_SIZE = 4

model_no_activation     = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=None).to(device)
model_relu              = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.ReLU()).to(device)
model_sigmoid           = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Sigmoid()).to(device)
model_tanh              = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Tanh()).to(device)
models = [model_no_activation, model_relu, model_sigmoid, model_tanh]

for m in models:
    # print_model_info(model=m)
    # plot_predictions(model=m, choose_class=False, add_to_title=f": activation function {m.activation}")
    plot_predictions(model=m, add_to_title=f": activation function {m.activation}")

# %% Exercise 3 & 4
# We use a single output variable for the binary classification, so we use BCE as a loss function

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

BATCH_SIZE = 64
TEST_SPLIT_FRACTION = 0.2
model = model_no_activation

# X_np, y_np = make_moons(n_samples=500, noise=0.15, random_state=0)
# X_np, y_np = make_blobs(n_samples=500, centers=2, n_features=2, random_state=0, cluster_std=0.5)
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
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Full dataset:\n\tExamples: {len(full_dataset)}\n\tShape of X: {X.shape}\n\tShape of y: {y.shape}")
print(f"Train/test dataloader:\n\tTrain data batches: {len(train_dataloader)}\n\tTest data batches: {len(test_dataloader)}")
print(f"\n\tTrain batches:")
for batch, (Xt, yt) in enumerate(train_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")
print(f"\n\tTest batches:")
for batch, (Xt, yt) in enumerate(test_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")

# %%

# logits = model(X)
# pred_prob = nn.Softmax(dim=1)(logits)[0].detach().cpu().numpy().tolist()
# y_classes = y.to(torch.long).flatten()

# nn.CrossEntropyLoss()(logits, y_classes)

# %%

test(test_dataloader, model)

# %%

def test(dataloader, model, verbose=True):

    total_loss = 0
    num_correct = 0
    num_samples = len(dataloader.dataset)

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:

            # unnormalized logits z, where sigmoid(z) or softmax(z) contains the class probabilities
            logits = model(X.to(device))

            if model.output_size == 1: # 1-dimensional binary classification
                y = y.to(device).to(torch.float32)
                total_loss += nn.BCEWithLogitsLoss(reduction='sum')(logits, y)
                num_correct += ((nn.Sigmoid()(logits) > 0.5) == y).sum()
            else: # multidimensional multiclass classification
                y = y.to(device).to(torch.long).flatten()
                total_loss += nn.CrossEntropyLoss()(logits, y).item() * len(X)
                num_correct += (nn.Softmax()(logits).argmax(dim=1) == y).sum().item()

    accuracy = num_correct / num_samples
    avg_loss = total_loss / num_samples 
    if verbose:
        print(f"Test Error:\n\tAccuracy: {100*accuracy:>.1f}% | Avg Loss: {avg_loss:>.8f}")
    
    return accuracy.item(), avg_loss.item()


def train(dataloader, model, loss_fn, optimizer, verbose=True):
    num_samples = len(dataloader.dataset)
    num_samples_processed = 0

    model.train()
    optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # make predictions
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward() # backpropagate gradients
        optimizer.step() # update weights
        optimizer.zero_grad() # reset gradients

        num_samples_processed += len(X)
        if batch % 3 == 0 and verbose:
            print(f"loss: {loss.item():>.7f}\t\t[processed {num_samples_processed:>5d}/{num_samples:>5d}]")

def grad_info(dataloader, model, loss_fn):

    mean_abs_grad = 0
    num_sat_weights = 0
    num_weights = 0
    model.train()
    model.zero_grad()

    X, y = next(iter(dataloader))

    loss = loss_fn(model(X), y)
    loss.backward()
    for name, param in model.named_parameters():
        mean_abs_grad += param.grad.abs().mean().item()
        num_weights += param.numel()
        num_sat_weights += (param.grad.abs() < 0.001).sum().item()
        
    frac_zero_grad = num_sat_weights / num_weights

    return mean_abs_grad, frac_zero_grad


# %%

model = SimpleMLP(hidden_size=16, activation=nn.ReLU()).to(device)
# plot_predictions(model, choose_class=True)
# plot_predictions(model, choose_class=False)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # must be instantiated AFTER the model

model(X)

# %%

EPOCHS = 1024
test(test_dataloader, model, loss_fn)
for t in range(EPOCHS):
    print(f"\nEpoch {t+1}\n-------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, verbose=False)
    test(test_dataloader, model, loss_fn)
    # mean_abs_grad, frac_zero_grad = grad_info(test_dataloader, model, loss_fn)
    # print(f"mean abs grad: {mean_abs_grad:>.5f}, frac zero grad: {frac_zero_grad:>.5f}")
print("Done!")

plot_predictions(model, choose_class=True)
# plot_predictions(model, choose_class=False)


# %%

loss_fn = nn.BCEWithLogitsLoss()
activations_list = [None, nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
epochs_list = [2 ** p for p in range(11)]
hidden_sizes_list = [2 ** p for p in range(10)]

for act in activations_list:
    for hid in hidden_sizes_list:
        print(f"ACTIVATION {act} | HIDDEN SIZE {hid}")

        model = SimpleMLP(hidden_size=hid, activation=act).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        acc, avg_loss = test(test_dataloader, model, loss_fn, verbose=False)
        print(f"\tepoch {0:>5d} | acc {100*acc:>3.1f}% | avg loss {avg_loss:>.7f}")
        for e in range(1,max(epochs_list)+1):
            train(train_dataloader, model, loss_fn, optimizer, verbose=False)
            if e in epochs_list:
                acc, avg_loss = test(test_dataloader, model, loss_fn, verbose=False)
                print(f"\tepoch {e:>5d} | acc {100*acc:>.1f}% | avg loss {avg_loss:>.7f}")
