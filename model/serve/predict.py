import numpy as np
import argparse
import json
import os
import pickle
import torch
import torch.optim as optim
import torch.utils.data

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


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    try:
        decoded = serialized_input_data.decode('utf-8')
        data = np.array(decoded.split(',')).astype(int)
        return data
    except:
        raise Exception('Failed to decode the string. Please send a plain CSV string.')


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)


def predict_fn(input_data, model):
    print('Predicting Image')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data = torch.from_numpy(np.array([input_data])).type(torch.FloatTensor)
        data = data.to(device)
    except:
        return 'Issue occured when transforming data to Tensor.'

    try:
        model.eval()
        with torch.no_grad():
            result = np.argmax(model.forward(data).numpy())
    except:
        return 'Issue occured when predicting.'

    return result