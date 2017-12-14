#k-Nearest Neighbors (kNN)

## Initialize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
fullFilePath = join(filePath,userDataSet)

#Import and Explore the Dataset
names = ['userID', 'movieID', 'rating', 'timestamp']
data = pd.read_table(fullFilePath, names=names).drop('timestamp', axis=1)
N = len(data)
print(data.shape)
print(data.head(2))

## Number of users:
Nu = len(data.userID.unique())
print(Nu)


## Number of movies:
Nm = len(data.movieID.unique())
print(Nm)


## Rating distribution:
data.rating.value_counts(sort=False).plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


## Number of movies per users:
movies_per_users = data.userID.value_counts()
print('min = %d, mean = %d, max = %d' %(movies_per_users.min(),
                                        movies_per_users.mean(),
                                        movies_per_users.max()))


## Number of users per movies:
users_per_movies = data.movieID.value_counts()
print('min = %d, mean = %d, max = %d' %(users_per_movies.min(),
                                        users_per_movies.mean(),
                                        users_per_movies.max()))


## Convert data to matrix
rating = data.pivot(index='userID', columns='movieID').rating
userID = rating.index
movieID = rating.columns
print(rating.shape)
print(rating.head(2))

## Evaluation metric: root-mean-squared error (RMSE)
def rmse(y, yhat):
    e2 = (y - yhat) ** 2
    return np.sqrt(e2.mean())


## Calculate pairwise distances between users
def dist(i,j):
    ## Only consider users who have at least 3 movies in common
    d = (rating.ix[i] - rating.ix[j])**2
    if d.count() >= 3:
        return np.sqrt(d.mean())

## Distance matrix
D = np.empty((Nu,Nu))
D.fill(np.nan)
D = pd.DataFrame(D)
D.index = D.columns = userID
for user1 in userID:
    for user2 in userID[userID > user1]:
        D[user1].ix[user2] = dist(user1, user2)
        D[user2].ix[user1] = D[user1].ix[user2]


## Predict ratings by the average of nearest neighbors
def kNN_predict(user, k=5):
    ## Sort users by distance
    neighbors = D[user].sort_values().index

    ## Function for calculting the kNN average
    def kNN_average(x):
        return x.ix[neighbors].dropna().head(k).mean()

    ## Apply kNN_average to every movie
    pred = rating.apply(kNN_average, axis=0)
    return pred


## Optimize k
user = 1
K = [3, 5, 10, 15]
error = []
for k in K:
    y = rating.ix[user]
    yhat = kNN_predict(user, k=k)
    err = rmse(y, yhat)
    error.append(err)
    print(k, '\t', err)

plt.plot(K, error, '-o')
plt.xlabel('# Nearest Neighbors')
plt.ylabel('Error (RMSE)')
plt.show()


## k = 5 gives the lowest error.

## Plot the predictions
def jitter(y, yhat, title=''):
    pred = np.round(yhat)
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    def noise():
        return np.random.randn(len(y)) * 0.1
    plt.scatter(y + noise(), pred + noise())
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(title)
    plt.axis('image')
    plt.grid()
    plt.show()
    print('error (rmse) =', rmse(y, yhat))

## Plot the predictions
user = 1
pred = kNN_predict(user, k=5)
jitter(rating.ix[user], pred, 'kNN')