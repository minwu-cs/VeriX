import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from verix_temperature_scaling.temperature_scaling import ModelWithTemperature


saved_data_path = './saved_data/'
output_path = './gtsrb_outputs/'
gtsrb_path = './train_networks/gtsrb.pickle'

def get_gtsrb_label(index):
    get_gtsrb_labels = ['50 mph', '30 mph', 'yield', 'priority road',
                        'keep right', 'no passing for large vechicles', '70 mph', '80 mph',
                        'road work', 'no passing']
    return get_gtsrb_labels[index]

def softmax(logits):
  return np.exp(np.max(logits, axis=1)) / np.sum(np.exp(logits), axis=1)

with open(gtsrb_path, 'rb') as handle:
  gtsrb = pickle.load(handle)
x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
x_test = x_test / 255
x_test_tensor = torch.tensor(x_test).to(torch.float32)
y_test_tensor = torch.tensor(y_test)

explanation_sizes = pd.read_pickle(saved_data_path + 'gtsrb-10x2-new-0.005-sizes')
explanation_sizes /= 32*32
explanation_sizes = torch.tensor(explanation_sizes)

logits_gtsrb = torch_model(x_test_tensor.flatten(1)).detach().numpy()
softmax_gtsrb = softmax(logits_gtsrb)
preds_gtsrb = np.argmax(logits_gtsrb, axis=1)
accuracy = sum(((preds_gtsrb - y_test) == 0) / y_test.shape[0])
print(f'accuracy = {accuracy}')

exit()

print('\nVanilla temp scaling')

test_dataset = TensorDataset(x_test_tensor.flatten(1)[:num_to_load], y_test_tensor[:num_to_load])

# Create a DataLoader
batch_size = 64  # Set your desired batch size
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                             sampler=SubsetRandomSampler(range(num_to_load)))

temp_scaled_model = ModelWithTemperature(torch_model)
temp_scaled_model.set_temperature(test_dataloader, 2000)

temp_scaled_model = ModelWithTemperature(torch_model)
temp_scaled_model.test_const_temperatures(test_dataloader, 0.5, 1.5, 0.01)