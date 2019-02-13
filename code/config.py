# Author: Hamid Karimi (karimiha@msu.edu)
# This file includes configurations of the project

import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def path(p):
    return os.path.expanduser(p)


parser = argparse.ArgumentParser(description='Arguments for Compromised Account Detection Project')

# The path where the project is
PATH = 'INSERT A PATH'

# The location of the project (e.g., data, features, etc)
parser.add_argument("--project_dir", type=path, required=False, default=PATH,
                    help="Directory containing the project files")

# The location where you want to save the models and results
parser.add_argument("--save_path", type=path, required=False, default=PATH + 'Models/',
                    help="Directory containing the saved model")

# The loation of tweet-level doc2vec. Theya are retrived and added to the other tweet information such as hashtag, mention, etc
parser.add_argument("--doc2vec_tweet_level_dir", type=str, required=False, default=PATH + 'Data/TweetsDoc2Vec/')

# The maximum sequence length of tweet features in our dataset is 1165. Pleaee do not change this for the current dataset
parser.add_argument("--max_seq_length", type=int, required=False, default=1165)

# Use this argument to differentiate between different experiments
parser.add_argument("--sim_name", type=str, required=False, default="sim1",
                    help="The unique and arbitrary name for the simulation")

# The hidden size in lstm  handling the sequence of tweet features
parser.add_argument("--lstm_hidden_size", type=int, required=False, default=200, help="The LSTM hidden size")

# The number of lstm layer  handling the sequence of tweet features
parser.add_argument("--lstm_num_layers", type=int, required=False, default=1, help="The number of LSTM layers")

# The output size of the fully connected layer used to tune the network features
parser.add_argument("--network_out_size", type=int, required=False, default=10,
                    help="The output size of the FC layer tuning the network features")

# The output size of the fully connected layer used to tune the user context features
parser.add_argument("--user_context_out_size", type=int, required=False, default=50,
                    help="The output size of the FC layer tuning the user context features")

# The ratio of the dropout to prevent overfitting (it must be between 0.0 and 1.0)
parser.add_argument("--dropout", type=float, required=False, default=0.3, help="The dropout")

# The ratio of the weight decay to prevent overfitting (i.e., L2 regularization)
parser.add_argument("--weight_decay", type=float, required=False, default=0.01, help="The dropout")

# A pretrained model can be loaded by setting this flag to true.
# Note that the model (named E2ECAD.model) should exist in args.save_path + args.sim_name + '/'
parser.add_argument("--load_pretrained", type=str2bool, required=False, default=False,
                    help="If true the program attempts to load the saved model")

# The size of fully connected network applied to the concatenation of all extracted featuer embeddings
parser.add_argument("--final_fc_output_size", type=int, required=False, default=30,
                    help="The size of final fully connected network")

# The mask used to mask different features.
# Features include namely Coordinate, Language, Place, Time, Media, Media,
# Retweet, URL, Hashtag, Mention, Sensitive, Source, Tweet, User, and Network
# Use the name of any of these features to mask (either lowercase or uppercase) separated by ';'
# For instance by setting mask='Hashtag, Coordinate, User', Hashtag, Coordinate,
# and User context features will be all masked to zero in the experiments
parser.add_argument("--mask", type=str, required=False, default="", help="Feature masks")

# The number of users (accounts) sampled at each iteration for optimizing the model.
# Note that batch_size /2 users are sampled from each class (i.e., compromised and not compromised)
parser.add_argument("--batch_size", type=int, required=False, default=128,
                    help="The batch size")

# The flag to use GPU (if available)
parser.add_argument('--gpu', type=str2bool, default='True', help='enables training on CUDA')

# The learning rate of updating the model parameters
parser.add_argument("--lr", type=float, required=False, default=0.01, help="Learning rate")

# The decay in learning rate of updating the model parameters
parser.add_argument("--decay_lr", type=float, required=False, default=0.95, help="The learning rate decay rate")

# The step size to decay the learning rate
parser.add_argument("--step_size_lr", type=float, required=False, default=20, help="The learning rate decay rate")

# The number of iterations to train the model
parser.add_argument("--iterations", type=int, required=False, default=200, help="The number of iterations")

args = parser.parse_args()
