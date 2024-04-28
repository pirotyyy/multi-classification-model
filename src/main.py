from dataset import load_data
from model import BlobModel
from trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    # set random seed
    torch.manual_seed(42)

    # load dataset
    X_train, X_test, y_train, y_test = load_data()

    # create model
    model = BlobModel(input_features=2, output_features=4, hidden_units=8)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    trainer = Trainer(model, criterion, optimizer)

    trainer.train(X_train, y_train, X_test, y_test, "cpu")


if __name__ == "__main__":
    main()
