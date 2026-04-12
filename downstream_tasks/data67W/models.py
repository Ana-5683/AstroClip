
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



def few_shot(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
) -> ndarray:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = MLP(
        n_in=X_train.shape[1],
        n_out=num_features,
        n_hidden=hidden_dims,
        act=[nn.ReLU()] * (len(hidden_dims) + 1),
        dropout=0.1,
    )

    # Set up the model
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda()).squeeze()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 输出每个epoch的平均loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.6f}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds


def zero_shot(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> ndarray:
    """Train a zero-shot model using KNN"""
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=64)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    return preds


class MLP(nn.Sequential):
    """MLP model"""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 2):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(n_[-2], n_[-1]))
        super(MLP, self).__init__(*layer)