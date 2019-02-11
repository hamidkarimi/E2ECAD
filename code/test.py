# Author: Hamid Karimi (karimiha@msu.edu)
# Use this file to get the performance of the model on the test set

import os
import numpy as np
import torch.utils.data
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

import config
import utils

args = config.args

cuda_enabled = args.gpu and torch.cuda.is_available()
device = torch.device("cuda" if cuda_enabled else "cpu")
print("The device is {}".format(device))

save_path = args.save_path + args.sim_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)  # Create a directory for saving the model and results

log_file = open(save_path + 'test.txt', 'w')

# Loading up the previous if exists and desired, or create a new model otherwise
if os.path.exists(save_path + 'E2ECAD.model'):
    model = torch.load(save_path + 'E2ECAD.model')
    print("A pre trained model loaded from {}".format(save_path))
else:
    print("ERROR. No trained model")
    exit(-1)

model.eval()
mask = utils.get_mask()
softmax = nn.Softmax(dim=1)

print('=' * 100)
X_tweet_features, X_network_features, X_user_context_features, seq_lengths, Y = \
    utils.get_batch(size_comp=-1, size_notcom=-1, mask=mask,
                    shuffle=False, split='test')
all_ouputs = []
for i in range(0, len(X_tweet_features), args.batch_size):
    start = i
    end = min(start + args.batch_size, len(X_tweet_features))
    outputs = model(torch.from_numpy(X_tweet_features[start:end]).float().to(device),
                    torch.from_numpy(X_network_features[start:end]).float().to(device),
                    torch.from_numpy(X_user_context_features[start:end]).float().to(device),
                    seq_lengths[start:end])
    all_ouputs.append(outputs)

all_ouputs = torch.cat(all_ouputs, dim=0)
prediction_probs = softmax(all_ouputs.detach()).cpu().numpy()

predictions = np.argmax(prediction_probs, axis=1)

_report = classification_report(y_true=Y, y_pred=predictions, target_names=['NotCompromised', 'Compromised'])
_confusion_matrix = confusion_matrix(y_true=Y, y_pred=predictions, labels=[0, 1])
_precision, _recal, _f1 = _report.split('\n')[3].split()[1], _report.split('\n')[3].split()[2], \
                          _report.split('\n')[3].split()[3]

print("F1 {} Precision {} Recall {}  ".format(_f1, _precision, _recal))
print(_report)
print(_confusion_matrix)
print('=' * 100)

log_file.write("F1 {} Precision {} Recall {} \n".format(_f1, _precision, _recal))
log_file.write("{}\n".format(_report))
log_file.write("{}\n\n".format(_confusion_matrix))
