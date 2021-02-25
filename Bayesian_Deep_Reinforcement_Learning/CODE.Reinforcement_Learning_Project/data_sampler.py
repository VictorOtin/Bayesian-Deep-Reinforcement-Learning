from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf


def sample_jester(file_name, context_dim, num_actions, num_contexts,
                        shuffle_rows=True, shuffle_cols=False):

    # with tf.gfile.Open(file_name, 'rb') as f:
    #   dataset = np.load(f)
  dataset=pd.read_csv(file_name)
  dataset = dataset.iloc[:num_contexts, :].to_numpy()

  # if shuffle_cols:
  #   dataset = dataset[:, np.random.permutation(dataset.shape[1])]
  # if shuffle_rows:
  #   np.random.shuffle(dataset)
   
  assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

  opt_actions = np.argmax(dataset[:, context_dim:], axis=1) #Devuelve el argumento en el que se da el valor maximo
  opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])
  return dataset, (opt_rewards, opt_actions)

def sample_EachMovies(file_name, context_dim, num_actions, num_contexts,
                        shuffle_rows=True, shuffle_cols=False):

    #Use ratings data to downsample tags data to only movies with ratings 
    with tf.gfile.Open(file_name, 'r') as f:
        Ratings = pd.read_csv(f, header=0, sep=' ',
                         na_values=[' ?']).dropna()
    Ratings.to_numpy()
    
    #Use ratings data to downsample tags data to only movies with ratings 
    
    Mean = Ratings.groupby(by="userId",as_index=False)['rating'].mean()
    Rating_avg = pd.merge(Ratings,Mean,on='userId')
    Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']
    # print(Rating_avg.head())
    # check = pd.pivot_table(Rating_avg,values='rating_x',index='userId',columns='movieId')
    final = pd.pivot_table(Rating_avg,values='adg_rating',index='userId',columns='movieId')
    
    # Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    dataset = final_user.to_numpy()
    
    dataset = dataset[:num_contexts, :]
    
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1) #Devuelve el argumento en el que se da el valor maximo
    opt_rewards = np.array([dataset[i, context_dim + a]
                              for i, a in enumerate(opt_actions)])
    return dataset, (opt_rewards, opt_actions)

def sample_MovieLens(file_name, context_dim, num_actions, num_contexts,
                        shuffle_rows=True, shuffle_cols=False):

    #Use ratings data to downsample tags data to only movies with ratings 
    with tf.gfile.Open(file_name, 'r') as f:
        Ratings = pd.read_csv(f, header=0,sep='\t',
                         na_values=[' ?']).dropna()
    Ratings.to_numpy()
    
    #Use ratings data to downsample tags data to only movies with ratings 
    
    Mean = Ratings.groupby(by="userId",as_index=False)['rating'].mean()
    Rating_avg = pd.merge(Ratings,Mean,on='userId')
    Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']
    # print(Rating_avg.head())
    # check = pd.pivot_table(Rating_avg,values='rating_x',index='userId',columns='movieId')
    final = pd.pivot_table(Rating_avg,values='adg_rating',index='userId',columns='movieId')
    
    # Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    dataset = final_user.to_numpy()
    
    dataset = dataset[:num_contexts, :]
    
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1) #Devuelve el argumento en el que se da el valor maximo
    opt_rewards = np.array([dataset[i, context_dim + a]
                              for i, a in enumerate(opt_actions)])
    return dataset, (opt_rewards, opt_actions)

def sample_modcloth(file_name, context_dim, num_actions, num_contexts,
                        shuffle_rows=True, shuffle_cols=False):

    #Use ratings data to downsample tags data to only movies with ratings 
    Ratings = pd.read_csv(file_name)
    
    Mean = Ratings.groupby(by="user_id",as_index=False)['rating'].mean()
    Rating_avg = pd.merge(Ratings,Mean,on='user_id')
    Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']
    # print(Rating_avg.head())
    # check = pd.pivot_table(Rating_avg,values='rating_x',index='userId',columns='movieId')
    final = pd.pivot_table(Rating_avg,values='adg_rating',index='user_id',columns='item_id')
    
    # Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    dataset = final_user.to_numpy()
    
    dataset = dataset[:num_contexts, :]
    
    # if shuffle_cols:
    #     dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    # if shuffle_rows:
    #     np.random.shuffle(dataset)
    
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1) #Devuelve el argumento en el que se da el valor maximo
    opt_rewards = np.array([dataset[i, context_dim + a]
                              for i, a in enumerate(opt_actions)])
    return dataset, (opt_rewards, opt_actions)
