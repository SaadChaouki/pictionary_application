import argparse
import json
import os
import pickle
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from classifier import NeuralNetClassifier


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetClassifier(model_info['input_size'],
                                model_info['hidden_size'],
                                model_info['output_size'],
                                model_info['dropout'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def _getLoaderTrainingJob(batchSize, dataFolder):
    print('Getting the training loader')
    trainingData = pickle.load(open(dataFolder + '/train.txt', 'rb'))
    train_y = torch.from_numpy(trainingData[0]).type(torch.FloatTensor)
    train_X = torch.from_numpy(trainingData[1]).type(torch.FloatTensor)
    trainDataset = torch.utils.data.TensorDataset(train_X, train_y)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize)
    return trainLoader


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    print("Initating Model Training.")
    model.to(device)

    # Training
    for epoch in range(epochs):
        # Setting the model to training mode.
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            # Changing the  inputs to the device used.
            inputs, labels = inputs.to(device), labels.to(device)

            # Setting gradients to 0
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Computing Loss
            running_loss += loss.item()
        # Tracking
        print("Epoch: {}, BCELoss: {}".format(epoch + 1, running_loss / len(train_loader)))

    return model


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')

    # Model Parameters
    parser.add_argument('--input_size', type=int, default=784, metavar='N')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N')
    parser.add_argument('--hidden_size', type=int, default=500, metavar='N')
    parser.add_argument('--output_size', type=int)

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # Load the training data.
    train_loader = _getLoaderTrainingJob(args.batch_size, args.data_dir)

    # Build the model.
    model = NeuralNetClassifier(args.input_size, args.hidden_size, args.output_size, args.dropout).to(device)

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'output_size': args.output_size,
            'dropout': args.dropout
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)


