# This file includes two classes needed to hold the data

import numpy as np
class Tweet():
    def __init__(self):
        self.id = 0  # ID of the tweet
        self.hashtag = np.zeros(29)  # Hashtages used e.g., #MSU
        self.coordinate = np.zeros(2)  # Longitude and Latitude
        self.media = 0  # Is media has been used in the tweet (e.g., video)
        self.mention = np.zeros(23)  # The mentions in the @hamidkarimi65
        self.retweet = 0  # Is that a retweet or not?
        self.sensitive = 0  # Does contain any sensitive words like
        self.source = 0  # What the source of the tweet e.g., Windowns OS
        self.language = 0  # The code for used languge
        self.time = 0  # Epoch time
        self.place = 0  # The place e.g., city
        self.url = 0  # Does it contain any URL?
        self.doc2vec_tweet = np.zeros(400)  # The doc2vec vector of the tweet's text

    def __getitem__(self, item):
        x = np.zeros(462)
        x[0:29] = self.hashtag
        x[29:31] = self.coordinate
        x[31] = self.media
        x[32:55] = self.mention
        x[55] = self.retweet
        x[56] = self.sensitive
        x[57] = self.source
        x[58] = self.language
        x[59] = self.time
        x[60] = self.place
        x[61] = self.url
        x[62:] = self.doc2vec_tweet
        return x

    def __str__(self):
        return "ID: {},#: {},LatLong: {}, media: {}, mention: {}, RT:{}, " \
               "sens: {}, src: {}, lang: {}, time:{}, place:{}, url: {} doc2vec Tweet {}". \
            format(self.tweetID, self.hashtags,
                   self.coordination, self.media,
                   self.mention, self.retweet, self.sensitive,
                   self.source, self.language, self.time,
                   self.place, self.url, self.doc2vec_tweet)

# This calss holds the features asscoaited with an user (account) --namely Tweet, Network, and User features.  (see TABLE I in E2ECAD paper)
class User():
    def __init__(self, ID ):
        self.ID = ID  # ID of the user (this is the internal ID not Twitter handle or ID)
        self.tweet_features = None  # A collection of Tweet featutes
        self.network_features = None  # Network features
        self.user_context_features = None # User context features
        self.tweet_features_seq_length = 0  # The sequence of tweet features are padded, so this filed keeps the length before padding

    def get_tweet_features(self, mask=np.ones(462)):
        return np.array([mask * t[0] for t in self.tweet_features])  # t[0] calls __getitem__ of Tweet class

    def get_network_features(self, mask=np.ones(12)):
        return mask * self.network_features

    def get_user_features(self, mask=np.ones(400)):
        return mask * self.user_context_features
