import os
import timeit
import pathlib
import torch

from torchinfo import summary
from torchmetrics import Accuracy
from dexnet.classes.PokemonClassifierModel import PokemonClassifierModel
from dexnet.data_load import create_dataloaders
from dexnet.dexio import save_weights, load_weights
from dexnet.utils.device_check import get_best_device


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
    # 1. Load data
    device = get_best_device()

    data_path = pathlib.Path(data_directory)
    train_dataloader, test_dataloader = create_dataloaders(data_path)

    model = PokemonClassifierModel()
    models_path = pathlib.Path("models/")

    # Load weights
    if os.listdir(models_path):
        model.load_state_dict(load_weights())
        print(f"Model weights loaded")

    model.to(device)

    # Summary
    summary(model=model, input_size=(1, 3, 64, 64))

    # 2. Loss, optimizer and metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           threshold=10e-4)
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

            scheduler.step(loss)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

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

        print(f"Epoch {epoch+1}:",
              f"train - acc = {train_acc:.5f}, loss = {train_loss:.5f}",
              f"| test - acc = {test_acc:.5f}, loss = {test_loss:.5f}",
              sep=" ")

        if (epoch+1) % 100 == 0:
            save_weights(model)

    end_time = timeit.default_timer()
    print(f"Train finished in {end_time - start_time} seconds.")


if __name__ == "__main__":
    train("../data/pokemons/", epochs=1000)
