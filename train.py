# Author: Hamid Karimi (karimiha@msu.edu)

# Use this file to train the model

import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import config
import utils
from model import E2ECAD

args = config.args
################ Initial setup ##################################

cuda_enabled = args.gpu and torch.cuda.is_available()
device = torch.device("cuda" if cuda_enabled else "cpu")
print("The device is {}".format(device))

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

save_path = args.save_path + args.sim_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)  # Create a directory for saving the model and results

with open(save_path + 'config', 'w') as config_file:
    argss = (str(args).split('(')[1].split(')')[0].split(','))
    for a in argss:
        config_file.write("{}\n".format(a))
    config_file.write("saved_ptah: {}".format(save_path))

log_file = open(save_path + 'log.txt', 'w')

# Loading up the previous if exists and desired, or create a new model otherwise
if os.path.exists(save_path + 'E2ECAD.model') and args.load_pretrained:
    model = torch.load(save_path + 'E2ECAD.model').to(device)
    print("A pre trained model loaded from {}".format(save_path))
else:
    model = E2ECAD().to(device)

params = [p for p in model.parameters() if p.requires_grad]
criterion = nn.CrossEntropyLoss()

# Dynamically changing the learning rate
optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=args.step_size_lr,
                   gamma=args.decay_lr)
# Getting the mask
mask = utils.get_mask()
softmax = nn.Softmax(dim=1)


################ The end of initial setup ########################

def run_simulation():
    def train():
        model.train()

        X_tweet_features, X_network_features, X_user_context_features, seq_lengths, Y = \
            utils.get_batch(size_comp=int(args.batch_size * (1 / 2)), size_notcom=int(args.batch_size * (1 / 2)),
                            mask=mask,
                            shuffle=True, split='train')

        prev_f1 = 0
        for iteration in range(args.iterations):

            # Passing the features and seq lengths to the models
            outputs = model(torch.from_numpy(X_tweet_features).float().to(device),
                            torch.from_numpy(X_network_features).float().to(device),
                            torch.from_numpy(X_user_context_features).float().to(device),
                            seq_lengths)

            # computing loss and back propagating error
            loss = criterion(outputs, torch.from_numpy(Y).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("iteration {} Loss {}".format(iteration, loss.detach().cpu().numpy()))

            # If there is an improvement, the current model is saved
            if iteration % 20 == 0 and iteration:
                model.eval()  # Setting the model in the eval model while evaluating
                current_f1 = eval(iteration)
                if current_f1 >= prev_f1:
                    prev_f1 = current_f1
                    torch.save(model, save_path + 'E2ECAD.model')
                    print("**** Best model saved F1 {} at iteration {}".format(current_f1, iteration))
                model.train()

    # Evaluating the model on the eval set
    def eval(iteration):
        print('=' * 100)
        log_file.write("Evaluation. iteration ({})    \n".format(iteration))
        print("      Evaluation. iteration ({})    ".format(iteration))
        X_tweet_features, X_network_features, X_user_context_features, seq_lengths, Y = \
            utils.get_batch(size_comp=-1, size_notcom=-1, mask=mask,
                            shuffle=False, split='eval')
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
        return float(_f1)

    train()
    log_file.close()


run_simulation()
