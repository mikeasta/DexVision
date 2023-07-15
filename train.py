import os
import timeit
import pathlib
import torch
import torchvision

from torchinfo import summary
from torchmetrics import Accuracy
from model.PokemonClassifierModel import PokemonClassifierModel
from data_loader.data_load import create_dataloaders
from utils.dexio import save_weights, load_weights
from utils import get_best_device
from utils.seed import set_seed


def train(
        data_directory: str,
        epochs: int = 50
) -> None:
    """
    Train baseline for PokemonClassifierModel.

    :param data_directory: Directory, from which images would be loaded
    :param epochs: Amount of epochs
    :return:
    """
    print(f"Train process started:",
          f"="*80,
          f"- Torch version: {torch.__version__}.",
          f"- Torchvision version: {torchvision.__version__}.",
          f"- Using device: {get_best_device()}.",
          f"- Model: {PokemonClassifierModel().__class__.__name__}.",
          f"- Train/test data path: {data_directory}.",
          f"- Epochs: {epochs}.",
          f"="*80,
          sep="\n")
    # 1. Load data
    device = get_best_device()

    data_path = pathlib.Path(data_directory)
    train_dataloader, test_dataloader = create_dataloaders(data_path)

    set_seed()
    model = PokemonClassifierModel()
    models_path = pathlib.Path("models/")

    # Load weights
    if models_path.is_dir():
        if os.listdir(models_path):
            model.load_state_dict(load_weights())
            print(f"Model weights for training loaded")

    model.to(device)

    # Summary
    summary(model=model, input_size=(1, 3, 64, 64))

    # 2. Loss, optimizer and metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min")
    accuracy_fn = Accuracy(task="multiclass", num_classes=4)

    # 3. Train loop
    start_time = timeit.default_timer()
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_predicted = model(X)
            loss = loss_fn(y_predicted, y)
            train_loss += loss
            train_acc += accuracy_fn(y_predicted.argmax(dim=1), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        scheduler.step(train_loss)

        model.eval()
        with torch.inference_mode():
            test_loss, test_acc = 0, 0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)

                y_predicted = model(X)
                loss = loss_fn(y_predicted, y)
                test_loss += loss
                test_acc += accuracy_fn(y_predicted.argmax(dim=1), y)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Epoch {epoch+1}, learning rate = {optimizer.param_groups[0]['lr']}:",
              f"train - acc = {train_acc:.5f}, loss = {train_loss:.5f}",
              f"| test - acc = {test_acc:.5f}, loss = {test_loss:.5f}",
              sep=" ")

        if (epoch+1) % 50 == 0:
            save_weights(model)

    end_time = timeit.default_timer()
    print(f"Train finished in {end_time - start_time} seconds.")


if __name__ == "__main__":
    train("../data/pokemons/", epochs=500)
