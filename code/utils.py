# Author: Hamid Karimi (karimiha@msu.edu)

import pandas as pd
import random
import config
import numpy as np
import pickle
from sklearn import preprocessing
from classes import Tweet, User

args = config.args


# This function calculates the clustering coefficients according to
# the Equation (2) of E2ECAD paper (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8508296)
def calculate_clustering_coeffiencts(matrix, connections):
    if len(connections) <= 1:
        return 0, 0, 0
    sum = 0
    for i in range(len(connections) - 1):
        for j in range(i + 1, (len(connections)), 1):
            connected = matrix[connections[j]][connections[i]][0]
            if connected == 1:
                sum = sum + 1
    cluster_coeff1 = 2.0 * sum / (1.0 * len(connections) * (len(connections) - 1))

    sum = 0
    for i in range(len(connections) - 1):
        for j in range(i + 1, (len(connections)), 1):
            connected = matrix[connections[j]][connections[i]][1]
            if connected == 1:
                sum = sum + 1

    cluster_coeff2 = 2.0 * sum / (1.0 * len(connections) * (len(connections) - 1))

    sum = 0
    for i in range(len(connections) - 1):
        for j in range(i + 1, (len(connections)), 1):
            connected = matrix[connections[j]][connections[i]][2]
            if connected == 1:
                sum = sum + 1
    cluster_coeff3 = 2.0 * sum / (1.0 * len(connections) * (len(connections) - 1))

    return cluster_coeff1, cluster_coeff2, cluster_coeff3


# This functions extract tweet related features for a user
def extract_tweet_features_by_user(user_id):
    user_hashtag = pickle.load(open(args.project_dir + 'Data/Hashtag/' + user_id + '.pkl', "rb"))
    user_mention = pickle.load(open(args.project_dir + 'Data/Mention/' + user_id + '.pkl', "rb"))
    user_coordinate = pickle.load(open(args.project_dir + 'Data/Coordinate/' + user_id + '.pkl', "rb"))
    user_language = pickle.load(open(args.project_dir + 'Data/Language/' + user_id + '.pkl', "rb"))
    user_media = pickle.load(open(args.project_dir + 'Data/Media/' + user_id + '.pkl', "rb"))
    user_place = pickle.load(open(args.project_dir + 'Data/Place/' + user_id + '.pkl', "rb"))
    user_retweet = pickle.load(open(args.project_dir + 'Data/Retweet/' + user_id + '.pkl', "rb"))
    user_sensitive = pickle.load(open(args.project_dir + 'Data/Sensitive/' + user_id + '.pkl', "rb"))
    user_source = pickle.load(open(args.project_dir + 'Data/Source/' + user_id + '.pkl', "rb"))
    user_time = pickle.load(open(args.project_dir + 'Data/Time/' + user_id + '.pkl', "rb"))
    user_url = pickle.load(open(args.project_dir + 'Data/URL/' + user_id + '.pkl', "rb"))
    user_tweet_doc2vec = pickle.load(open(args.project_dir + 'Data/Doc2VecTweets/' + user_id + '.pkl', "rb"))

    # keys of each feature map is the (internal) ids assigned to tweets (incrementally from 1)
    ids = list(user_hashtag.keys()) + list(user_mention.keys()) + list(user_coordinate.keys()) + list(
        user_language.keys()) + list(user_media.keys()) + list(user_place.keys()) + list(user_retweet.keys()) + list(
        user_sensitive.keys()) + list(user_source.keys()) + list(user_time.keys()) + list(user_url.keys())

    seq_length = int(max([int(i) for i in ids]))  # The maximum id determines the length of the sequence

    user_tweets = []
    for i in range(1, args.max_seq_length + 1, 1):

        tweet_data = Tweet()
        tweet_data.id = i
        tweet_data.doc2vec_tweet = user_tweet_doc2vec[i - 1]

        if str(i) in user_hashtag.keys():
            tweet_data.hashtag[:len(user_hashtag[str(i)])] = user_hashtag[str(i)]

        if str(i) in user_mention.keys():
            tweet_data.mention[:len(user_mention[str(i)])] = user_mention[str(i)]

        if str(i) in user_coordinate.keys():
            tweet_data.coordinate = user_coordinate[str(i)]

        if str(i) in user_language.keys():
            tweet_data.language = user_language[str(i)][0]

        if str(i) in user_media.keys():
            tweet_data.media = user_media[str(i)][0]

        if str(i) in user_place.keys():
            tweet_data.place = user_place[str(i)][0]

        if str(i) in user_retweet.keys():
            tweet_data.retweet = user_retweet[str(i)][0]

        if str(i) in user_sensitive.keys():
            tweet_data.sensitive = user_sensitive[str(i)][0]

        if str(i) in user_source.keys():
            tweet_data.source = user_source[str(i)][0]

        if str(i) in user_time.keys():
            tweet_data.time = user_time[str(i)]

        if str(i) in user_url.keys():
            tweet_data.url = user_url[str(i)][0]

        user_tweets.append(tweet_data)
    return user_tweets, seq_length


def extract_network_features():
    id_label_split_records = pd.read_csv(args.project_dir + 'Data/ID-TwitterHandle-Label-Split.csv',
                                         sep=',', names=["ID", "TwitterHandle", "Label", "Split"])

    # We need to load the adjacency matrix
    adjacency_matrix = np.array(pickle.load(open(args.project_dir + 'Data/Network/adjacency_matrix.pkl', "rb")))

    id_label_split_records_file = open(args.project_dir + 'Data/ID-TwitterHandle-Label-Split.csv', 'r')
    id_label_split_records = [x.strip() for x in id_label_split_records_file.readlines()]
    id_label_split_records_file.close()

    all_network_features = []
    for i, row in enumerate(id_label_split_records):
        print("{}/{}".format(i, len(id_label_split_records)))
        record = np.concatenate([adjacency_matrix[i][0:i], adjacency_matrix[i][i + 1:]])
        #############################################
        ####### First set of network features #######
        #############################################

        mean = np.mean(record, axis=0)

        #############################################
        ####### Second set of network features #######
        #############################################

        # For reference to the different connections refer to the
        # Section IV.C of E2ECAD paper (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8508296) and especially Figure 3

        direct_connect = [i for i, r in enumerate(record) if r[0] == 1]  # Typer 1 connections
        indirect_connect_by_our_users = [i for i, r in enumerate(record) if r[1] == 1]  # Typer 2 connections
        indirect_connect_by_other_users = [i for i, r in enumerate(record) if r[2] == 1]  # Typer 3 connections

        # Calculating clustering coefficients
        c1, c2, c3 = calculate_clustering_coeffiencts(adjacency_matrix, direct_connect)
        c4, c5, c6 = calculate_clustering_coeffiencts(adjacency_matrix, indirect_connect_by_our_users)
        c7, c8, c9 = calculate_clustering_coeffiencts(adjacency_matrix, indirect_connect_by_other_users)

        all_network_features.append([mean[0], mean[1], mean[2], c1, c2, c3, c4, c5, c6, c7, c8, c9])
    all_network_features = np.array(all_network_features)
    norm_all_network_features = preprocessing.normalize(all_network_features)
    for i, row in enumerate(id_label_split_records):
        id = str(row.split(',')[0])
        with open(args.project_dir + 'Data/Network/' + id + '.pkl', 'wb') as f:
            pickle.dump(norm_all_network_features[i], f)


# This function creates aggregated pickle files ready to be fed into the model
def create_dataset():
    # First we need to load the file containing IDs, Labels, and Split. This file contains Twitter handles though we don't use it for training
    id_label_split_records = pd.read_csv(args.project_dir + 'Data/ID-TwitterHandle-Label-Split.csv',
                                         sep=',', names=["ID", "TwitterHandle", "Label", "Split"])

    user_context = pickle.load(open(args.project_dir + "/Data/UserContext.pkl", "rb"))
    # extract_network_features()

    for i, id in enumerate(id_label_split_records.ID):
        print("{}/{}".format(i, len(id_label_split_records)))
        user = User(ID=id)
        tweets, seq_length = extract_tweet_features_by_user(user_id=str(id))
        user.tweet_features = tweets
        user.tweet_features_seq_length = seq_length
        user.network_features = pickle.load(open(args.project_dir + 'Data/Network/' + str(id) + '.pkl', "rb"))
        user.user_context_features = user_context[int(id) - 1]
        with open(args.project_dir + 'Data/Dataset/' + str(id) + '.pkl', "wb") as f:
            pickle.dump(user, f)


# create_dataset() # You don't have to call this function as the dataset is already prepared

# size_comp and size_notcom specofies the number of Compromised and Not Compromised accounts, respectively.
# If they are -1 the entire set is selected
# Mask specifies which features to mask
# Shuffle for shuffling the batch (not be used during test or evaulation)
# Split determines which split to get the data
def get_batch(size_comp=64, size_notcom=64, mask=np.ones(874), shuffle=True, split='train'):
    id_label_split_records = pd.read_csv(args.project_dir + 'Data/ID-TwitterHandle-Label-Split.csv', sep=',',
                                         names=["ID", "TwitterHandle", "Label", "Split"])
    id_label_split_records = id_label_split_records[id_label_split_records.Split == split]

    id_label_split_records_comp = list(id_label_split_records[id_label_split_records.Label == 'Compromised'].ID.values)
    id_label_split_records_notcom = list(
        id_label_split_records[id_label_split_records.Label == 'Not Compromised'].ID.values)

    if size_comp == -1:
        size_comp = len(id_label_split_records_comp)
    if size_notcom == -1:
        size_notcom = len(id_label_split_records_notcom)
    if shuffle:
        random.shuffle(id_label_split_records_comp)
        random.shuffle(id_label_split_records_notcom)

    id_label_split_records_comp = id_label_split_records_comp[0:size_comp]
    id_label_split_records_notcom = id_label_split_records_notcom[0:size_notcom]

    Y = [1 for _ in range(size_comp)] + [0 for _ in range(size_notcom)]
    X_tweet_features, X_network_features, X_user_context_features = [], [], []
    seq_lengths = []

    for id in id_label_split_records_comp + id_label_split_records_notcom:
        user = pickle.load(open(args.project_dir + 'Data/Dataset/' + str(id) + '.pkl', 'rb'))
        X_tweet_features.append(user.get_tweet_features(mask[0:462]))
        X_network_features.append(user.get_network_features(mask[462:474]))
        X_user_context_features.append(user.get_user_features(mask[474:]))
        seq_lengths.append(user.tweet_features_seq_length)
    X_tweet_features, X_network_features, X_user_context_features = \
        np.array(X_tweet_features), np.array(X_network_features), np.array(X_user_context_features)

    seq_lengths = np.array(seq_lengths)
    Y = np.array(Y)
    return X_tweet_features, X_network_features, X_user_context_features, seq_lengths, Y


# This function returns the numerical value of the mask
def get_mask():
    mask = np.ones(874)  # By default, all features are included
    mask_string = args.mask
    mask_fields = mask_string.split(',')
    mask_fields = [x.lower() for x in mask_fields]

    if 'hashtag' in mask_fields:
        mask[0:29] = np.zeros(29)
    if 'coordinate' in mask_fields:
        mask[29:31] = np.zeros(2)
    if 'media' in mask_fields:
        mask[31] = 0
    if 'mention' in mask_fields:
        mask[32:55] = np.zeros(23)
    if 'retweet' in mask_fields:
        mask[55] = 0
    if 'sensitive' in mask_fields:
        mask[56] = 0
    if 'source' in mask_fields:
        mask[57] = 0
    if 'language' in mask_fields:
        mask[58] = 0
    if 'time' in mask_fields:
        mask[59] = 0
    if 'place' in mask_fields:
        mask[60] = 0
    if 'url' in mask_fields:
        mask[61] = 0
    if 'tweet' in mask_fields:
        mask[62:462] = np.zeros(400)
    if 'network' in mask_fields:
        mask[462:474] = np.zeros(12)
    if 'user' in mask_fields:
        mask[474:] = np.zeros(400)
    return mask
