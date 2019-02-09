# Author: Hamid Karimi (karimiha@msu.edu)

import torch
import torch.utils.data
from torch import nn

import config

args = config.args


class E2ECAD(nn.Module):
    def __init__(self):
        super(E2ECAD, self).__init__()
        # An LSTM layer to handle temporal information in the sequence of tweets
        self.lstm_tweet_feature_network = nn.LSTM(input_size=462, hidden_size=args.lstm_hidden_size,
                                                  num_layers=args.lstm_num_layers, batch_first=True,
                                                  dropout=args.dropout)

        # A fully connected layer to further tune the network features
        self.newtork_fc = nn.Sequential(nn.Linear(in_features=12, out_features=args.network_out_size),
                                        nn.ReLU())

        # The final fully connected layer
        self.FC = nn.Sequential(
            nn.Linear(args.lstm_hidden_size + args.network_out_size + 400, args.final_fc_output_size), nn.ReLU(),
            nn.Dropout(args.dropout))

        self.final_binary_classifier = nn.Linear(args.final_fc_output_size, 2)

    def forward(self, X_tweet_features, X_network_features, X_user_context_features, seq_lengths):
        seq_lengths = [s - 1 for s in seq_lengths]
        temp, _ = self.lstm_tweet_feature_network(X_tweet_features)

        # Getting the the embeddings corresponding to the last relevant element in the sequence
        tweet_embeddings = torch.stack([x[s, :] for x, s in zip(temp, seq_lengths)])
        network_embeddings = self.newtork_fc(X_network_features)
        final_embeddings = self.FC(torch.cat((network_embeddings, X_user_context_features, tweet_embeddings), dim=1))

        outputs = self.final_binary_classifier(final_embeddings)

        return outputs
