# %% Set up
import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")

# %% Generate data

NUM_CLASSES = 5

# X_np, y_np = make_moons(n_samples=500, noise=0.15, random_state=0)
X_np, y_np = make_blobs(n_samples=50, centers=NUM_CLASSES, n_features=2, random_state=0, cluster_std=1.5)
plt.scatter(X_np[:,0], X_np[:,1], c=y_np)

X = torch.tensor(X_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np).to(device)


# %% Create test and train data splits

BATCH_SIZE = 64
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2

# create train and test datasets
full_dataset = TensorDataset(X, y)
test_dataset, validation_dataset, subtrain_dataset = random_split(
    full_dataset,
    [TEST_FRACTION, VALIDATION_FRACTION, 1-TEST_FRACTION-VALIDATION_FRACTION],
    torch.Generator().manual_seed(42)
)
subtrain_dataloader = DataLoader(subtrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Full dataset:\n\tExamples: {len(full_dataset)}\n\tShape of X: {X.shape}\n\tShape of y: {y.shape}")
print(f"Dataloaders:\n\tSubtrain data batches: {len(subtrain_dataloader)}\n\tValidation data batches: {len(validation_dataloader)}\n\tTest data batches: {len(test_dataloader)}")
print(f"\n\tSubtrain batches:")
for batch, (Xt, yt) in enumerate(subtrain_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")
print(f"\n\tValidation batches:")
for batch, (Xt, yt) in enumerate(validation_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")
print(f"\n\tTest batches:")
for batch, (Xt, yt) in enumerate(test_dataloader):
    print(f"\tBatch {batch+1} | X: shape {tuple(Xt.shape)} | y: shape {tuple(yt.shape)}, dtype {yt.dtype}")

# %% Functions and classes for creating MLPs and plotting outputs and info

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=NUM_CLASSES, activation=None):
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
    # TODO: currently cannot output probabilities in e.g. the binary classification case

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
    model.eval()
    with torch.no_grad():
        grid_pred_class = nn.Softmax(dim=1)(model(grid_tensor)).argmax(dim=1) # pick the class with the highest prob
    grid_pred_class = grid_pred_class.detach().cpu().numpy()
    grid_pred_class = grid_pred_class.reshape(xx1.shape)

    cmap_to_use = plt.cm.tab10
    vmin_val = 0
    vmax_val = max(1, model.output_size-1)
    levels = np.arange(vmin_val - 0.5, vmax_val + 1.5, 1)

    plt.figure(figsize=(8,8))
    plt.contourf(xx1, xx2, grid_pred_class, cmap=cmap_to_use, alpha=0.8, 
                 vmin=vmin_val, vmax=vmax_val, levels=levels)
    plt.scatter(X_np[:,0], X_np[:,1], c=y_np, cmap=cmap_to_use, edgecolors='k', s=30,
                vmin=vmin_val, vmax=vmax_val)
    plt.title("Predicted classes and decision boundary" + add_to_title)

def print_model_info(model):
    print(f"Model structure:\n{model}\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | \n\tSize: {param.shape} | \n\tWeights for first unit in next layer : {param[:1]} \n")


# %% Instantiate a few different MLPs

HIDDEN_SIZE = 4

model_no_activation     = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=None).to(device)
model_relu              = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.ReLU()).to(device)
model_sigmoid           = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Sigmoid()).to(device)
model_tanh              = SimpleMLP(hidden_size=HIDDEN_SIZE, activation=nn.Tanh()).to(device)
models = [model_no_activation, model_relu, model_sigmoid, model_tanh]

for m in models:
    print_model_info(model=m)
    plot_predictions(model=m, add_to_title=f": activation function {m.activation}")


# %% Functions for training, testing and visualizing MLP performance

def test(dataloader, model, verbose=True):
    total_loss = num_correct = 0
    num_samples = len(dataloader.dataset)

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X) # unnormalized logits z, where softmax(z) gives the class probabilities
            total_loss += nn.CrossEntropyLoss(reduction='sum')(logits, y).item()
            num_correct += (nn.Softmax(dim=1)(logits).argmax(dim=1) == y).sum().item()

    accuracy = num_correct / num_samples
    avg_loss = total_loss / num_samples 
    if verbose:
        print(f"Test Error:\n\tAccuracy: {100*accuracy:>.1f}% | Avg Loss: {avg_loss:>.8f}\n")
    
    return accuracy, avg_loss

def train(dataloader, model, optimizer, verbose=True):
    num_samples_processed = 0
    num_samples = len(dataloader.dataset)

    model.train()
    optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # make prediction and compute loss
        logits = model(X)
        loss = nn.CrossEntropyLoss()(logits, y)

        # update parameters using SGD
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagate gradients
        optimizer.step() # update weights

        num_samples_processed += len(X)
        if batch % 2 == 0 and verbose:
            print(f"loss: {loss.item():>.7f}\t\t[processed {num_samples_processed:>5d}/{num_samples:>5d}]")

def compute_norms(model):
    param_l2_norms = {}
    grad_l2_norms = {}
    # param_names = []

    for name, param in model.named_parameters():

        # param_names.append(name)

        param_l2_norm = torch.norm(param.data, p=2).item()
        param_l2_norms[name] = param_l2_norm

        if param.grad is not None:
            grad_l2_norm = torch.norm(param.grad, p=2).item()
        else:
            grad_l2_norm = 0
        grad_l2_norms[name] = grad_l2_norm

    return param_l2_norms, grad_l2_norms

# %% Perform a single training run

EPOCHS = 1024

model = SimpleMLP(hidden_size=128, activation=nn.ReLU()).to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
    weight_decay=0
) # must be instantiated AFTER the model

train_acc, train_loss = test(subtrain_dataloader, model, verbose=False)
test_acc, test_loss = test(test_dataloader, model)
param_l2_norms, grad_l2_norms = compute_norms(model)
initial_logs = {
    'epoch': 0,
    'train_accuracy': train_acc,
    'train_loss': train_loss,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'param_norms': param_l2_norms,
    'grad_norms': grad_l2_norms
}

logs = []
logs.append(initial_logs)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}\n-------------------------------------")
    train(subtrain_dataloader, model, optimizer, verbose=False)

    train_acc, train_loss = test(subtrain_dataloader, model, verbose=False)
    test_acc, test_loss = test(test_dataloader, model)
    param_l2_norms, grad_l2_norms = compute_norms(model)
    log_entry = {
        'epoch': epoch+1,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'param_norms': param_l2_norms,
        'grad_norms': grad_l2_norms
    }
    logs.append(log_entry)
    
plot_predictions(model)

# %%

experiment_info = 'sgd without weight decay'


# def plot_training_metrics()
df_logs = pd.json_normalize(logs, sep='_').set_index('epoch')

acc_mask = [col for col in df_logs.columns if 'acc' in col]
loss_mask = [col for col in df_logs.columns if 'loss' in col]  
param_norm_mask = [col for col in df_logs.columns if 'param_norm' in col] 
grad_norm_mask = [col for col in df_logs.columns if 'grad_norm' in col] 

fig, axes = plt.subplots(4, 1, figsize=(8,32))
axes = axes.flatten()

axes[0].plot(df_logs[acc_mask], label=acc_mask)
axes[0].legend()
axes[0].set_title('accuracy')
axes[0].grid()

axes[1].plot(df_logs[loss_mask], label=loss_mask)
axes[1].legend()
axes[1].set_title('loss')
axes[1].grid()

axes[2].plot(df_logs[grad_norm_mask], label=grad_norm_mask)
axes[2].legend()
axes[2].set_title('gradient L2 norms')
axes[2].grid()

axes[3].plot(df_logs[param_norm_mask], label=param_norm_mask)
axes[3].legend()
axes[3].set_title('parameter L2 norms')
axes[3].grid()


# %%

# loss_fn = nn.BCEWithLogitsLoss()
activations_list = [None, nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
epochs_list = [2 ** p for p in range(11)]
hidden_sizes_list = [2 ** p for p in range(10)]

for act in activations_list:
    for hid in hidden_sizes_list:
        print(f"ACTIVATION {act} | HIDDEN SIZE {hid}")

        model = SimpleMLP(hidden_size=hid, activation=act).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        acc, avg_loss = test(test_dataloader, model, verbose=False)
        print(f"\tepoch {0:>5d} | acc {100*acc:>3.1f}% | avg loss {avg_loss:>.7f}")
        for e in range(1,max(epochs_list)+1):
            train(subtrain_dataloader, model, optimizer, verbose=False)
            if e in epochs_list:
                acc, avg_loss = test(test_dataloader, model, verbose=False)
                print(f"\tepoch {e:>5d} | acc {100*acc:>.1f}% | avg loss {avg_loss:>.7f}")
