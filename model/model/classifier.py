import torch.nn as nn
from collections import OrderedDict


class NeuralNetClassifier(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, dropout=0.0):
        super(NeuralNetClassifier, self).__init__()

        self.classifier = nn.Sequential(OrderedDict([

            ('linear1', nn.Linear(inputSize, hiddenSize)),
            ('relu1', nn.ReLU()),

            ('linear2', nn.Linear(hiddenSize, hiddenSize)),
            ('batchnorm1', nn.BatchNorm1d(num_features=hiddenSize)),
            ('relu2', nn.ReLU()),

            ('dropout1', nn.Dropout(dropout)),

            ('linear3', nn.Linear(hiddenSize, hiddenSize)),
            ('relu3', nn.ReLU()),

            ('linear4', nn.Linear(hiddenSize, hiddenSize)),
            ('batchnorm2', nn.BatchNorm1d(num_features=hiddenSize)),
            ('relu4', nn.ReLU()),

            ('linear5', nn.Linear(hiddenSize, outputSize))

        ])
        )

    def forward(self, x):
        x = self.classifier(x)
        return x