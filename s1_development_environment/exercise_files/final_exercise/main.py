import click
import torch
from model import MyAwesomeModel
import tqdm
from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        train_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch} - Training loss: {train_loss/len(trainloader)}")
    torch.save(model, "model.pt")

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            accuracy += (predicted == labels).sum().item()
    print(f"Accuracy: {accuracy/len(test_set)}")



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
