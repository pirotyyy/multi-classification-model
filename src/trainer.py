import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        device,
        n_epochs=100,
    ):
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        for epoch in range(n_epochs):
            ## Training
            # Set train-mode
            self.model.train()

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_train)

            # Calculate the loss
            loss = self.criterion(outputs, y_train)

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Calculate accuracy
            y_pred = torch.argmax(outputs, dim=1)
            train_acc = calc_acc(y_true=y_train, y_pred=y_pred)

            ## Testing
            self.model.eval()
            with torch.inference_mode():
                test_outputs = self.model(X_test)
                test_loss = self.criterion(test_outputs, y_test)

                # Calculate accuracy
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = calc_acc(y_true=y_test, y_pred=test_pred)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch} | Loss: {loss:.4f} Acc: {train_acc:.2f} | Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}"
                )


def calc_acc(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
