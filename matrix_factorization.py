#Importing the Libraries
import numpy as np
import pandas as pd
from os.path import join
import os
import DropboxAPI


#Fetching DataSet from DropBox  and Unzipping the File
url ='https://www.dropbox.com/sh/euppz607r6gsen2/AAAQCu8KjT7Ii1R60W2-Bm1Ua/MovieLens%20(Movie%20Ratings)?dl=1'
zipFileName = 'MovieLens (Movie Ratings).zip'
subzipFileName ='movielens100k/ml-100k'
userDataSet = 'u.data'
userTestDataSet = 'u1.test'
destPath = os.getcwd()
DropboxAPI.fetchData(url, zipFileName, destPath)
filePath = join(destPath, zipFileName.rsplit(".", 1)[0])
filePath = join(filePath,subzipFileName.rsplit(".", 1)[0])
#fullFilePath = join(filePath,userDataSet)

#Importing the Dataset
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(join(filePath,userDataSet), sep='\t', names=names)


#Calculating Number of Unique Users and Unique Movies
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]


# Create r_{ui}, our ratings matrix
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]


# Split into training and test sets.
# Remove 10 ratings for each user
# and assign them to the test set
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


train, test = train_test_split(ratings)
print('ratings shape', ratings.shape)

from sklearn.metrics import mean_squared_error


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def get_rmse(pred, actual):
    # Ignore nonzero terms.
    import math
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, actual))


from numpy.linalg import solve


class ExplicitMF():
    def __init__(self,
                 ratings,
                 n_factors=40,
                 learning='sgd',
                 item_fact_reg=0.0,
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        item_fact_reg : (float)
            Regularization term for item latent factors

        user_fact_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Regularization term for item biases

        user_bias_reg : (float)
            Regularization term for user biases

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            print
            'sample_row', self.sample_row
            print
            'sample_col', self.sample_col
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10, learning_rate=0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        # self.user_vecs = np.random.normal(scale=1./self.n_factors,\
        #                                  size=(self.n_users, self.n_factors))
        self.user_vecs = np.zeros(shape=(self.n_users, self.n_factors))
        # self.item_vecs = np.random.normal(scale=1./self.n_factors,
        #                                  size=(self.n_items, self.n_factors))
        self.item_vecs = np.zeros(shape=(self.n_items, self.n_factors))

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print
                '\tcurrent iteration: {:d}'.format(ctr)
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs,
                                               self.item_vecs,
                                               self.ratings,
                                               self.user_fact_reg,
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs,
                                               self.user_vecs,
                                               self.ratings,
                                               self.item_fact_reg,
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)  # error

            # Update biases
            self.user_bias[u] += self.learning_rate * \
                                 (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                 (e - self.item_bias_reg * self.item_bias[i])

            # Update latent factors
            self.user_vecs[u, :] += self.learning_rate * \
                                    (e * self.item_vecs[i, :] - \
                                     self.user_fact_reg * self.user_vecs[u, :])
            self.item_vecs[i, :] += self.learning_rate * \
                                    (e * self.user_vecs[u, :] - \
                                     self.item_fact_reg * self.item_vecs[i, :])

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            # print 'global_bias_user_bias_item_bias', prediction
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            # print 'prediction', prediction
            return prediction

    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        print('predictions_all shape', predictions.shape)
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
        # print predictions
        return predictions

    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).

        The function creates two new class attributes:

        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        self.train_rmse = []
        self.test_rmse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.train_rmse += [get_rmse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            self.test_rmse += [get_rmse(predictions, test)]
            if self._v:
                print
                'train mse: ' + str(self.train_mse[-1])
                print
                'Train rmse: ' + str(self.train_rmse[-1])
                print
                'Text mse: ' + str(self.test_mse[-1])
                print
                'Test rmse: ' + str(self.test_rmse[-1])
            iter_diff = n_iter


# MF_ALS = ExplicitMF(train, n_factors=40, \
#                    user_reg=0.0, item_reg=0.0)
# iter_array = [1, 2, 5, 10, 25, 50, 100]
# MF_ALS.calculate_learning_curve(iter_array, test)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse, \
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, \
             label='Test', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MSE', fontsize=30);
    plt.legend(loc='best', fontsize=20);


best_als_model = ExplicitMF(ratings, n_factors=20, learning='als', \
                            item_fact_reg=0.01, user_fact_reg=0.01, verbose=True)

iter_array = [50]

best_als_model.calculate_learning_curve(iter_array, test)

best_sgd_model = ExplicitMF(train, n_factors=80, learning='sgd', \
                            item_fact_reg=0.01, user_fact_reg=0.01, \
                            user_bias_reg=0.01, item_bias_reg=0.01, verbose=True)
# best_sgd_model.train(200, learning_rate=0.001)
iter_array = [200]
best_sgd_model.calculate_learning_curve(iter_array, test)

