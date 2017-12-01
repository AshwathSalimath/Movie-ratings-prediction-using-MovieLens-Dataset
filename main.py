
#Importing the Libraries
import numpy as np
import pandas as pd

#Importing the Dataset
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=names)

#Calculating Number of Unique Users and Unique Movies
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

#Creating a Ratings Matrix with size (n_users X n_items)
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

#Calculating minumum number of movies rated by each user
nonzero_counts = np.count_nonzero(ratings, axis=1)
print ('Number of minumum movies rated by each user : ', min(nonzero_counts))

#Calculating sparsity of ratings matrix
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('Sparsity of ratings matrix : ', sparsity)