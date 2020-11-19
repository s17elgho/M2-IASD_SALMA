import numpy as np
from keras.models import Model
from keras.layers import Dropout,Dense, Activation, Input, Lambda ## layers of the model
from tensorflow.keras.optimizers import SGD ## for learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential ## for building the model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from keras.layers import Conv2D
from keras.layers import Input, Lambda, Dense, Flatten,MaxPooling2D, concatenate
import random
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import pandas as pd
from keras import backend as K
import seaborn as sns
import plotly.express as px
from itertools import permutations
from keras.optimizers import SGD,Adam





def load_CIFAR10_data():
  ## import CIFAR-10
  (train_images, train_labels),(test_images, test_labels)= cifar10.load_data()

  ## nb class and shape of images
  nb_classes = 10
  nb_samples_train = train_images.shape[0]
  nb_samples_test = test_images.shape[0]
  img_rows, img_cols,channels = train_images.shape[1],train_images.shape[2],train_images.shape[3]

  ## flatten for MLP and normalize
  train = train_images.reshape(nb_samples_train, img_rows*img_cols*channels)
  test = test_images.reshape(nb_samples_test, img_rows*img_cols*channels)
  x_train = train.astype("float32") / 255
  x_test = test.astype("float32") / 255

  return x_train,train_labels,x_test,test_labels

img_size = 3072

## group by class
### all_train[7] contains all train images of class 7 
### all_test[7] contains all test images of class 7
def data_groupby_class(train_images,train_labels,test_images,test_labels):

  all_train = {}
  all_test = {}
  for classe in range(10):
    indexes_train = np.where(train_labels == classe)[0]
    indexes_test = np.where(test_labels == classe)[0]
    all_train[classe] = train_images[indexes_train,:]
    all_test[classe] = test_images[indexes_test,:]

  return all_train, all_test

### prepare pairs of same class using classes with more than one sample
def prepare_same_pairs(all_data,n_pairs_same,train_or_test):
  pairs_same = np.zeros((n_pairs_same,img_size,2))
  labels_same = np.zeros(n_pairs_same) ## keep track of the labels

  for i in range(n_pairs_same):
    ## for each time you want to create a pair of same class 
    ## first choose a class at random :  class  between 0 and 4 then randomly select two examples of that class from all_train[class]
    ## pairs_same[0,:,i] is the first example from the ith pair
    ## pairs_same[1,:,i] is the second example from the ith pair , they both are from same class
    if train_or_test == "train":
      classe = np.random.randint(0,5)
    elif train_or_test == "test" :
      classe = np.random.randint(0,10)
    else :
      raise NameError('Specify if train or train')
    row_ind1 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv1 = all_data[classe][row_ind1]
    row_ind2 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv2 = all_data[classe][row_ind2]
    label = classe
    labels_same[i] = label
    pairs_same[i,:,0] = indiv1
    pairs_same[i,:,1] = indiv2
  
  return (pairs_same,labels_same)


### prepare pairs of different class using classes with more than one sample

def prepare_different_pairs(all_data, n_pairs_diff,train_or_test):

  pairs_diff = np.zeros((n_pairs_diff,img_size,2))
  labels_diff = np.zeros((n_pairs_diff,2)) ## keep track of the labels 

  ## for each time you want to create a pair of different class 
  ## first choose two different classes at random :  classes_1 and classes_2  between 0 and 4 then randomly select one example from each classe from all_train[classes_1]
  ## and from all_train[classes_2]
  ## pairs_diff[0,:,i] is the first example from the ith pair
  ## pairs_diff[1,:,i] is the second example from the ith pair , they both are from different classes
  classes = [-1,-1]
  list_of_range = list(range(0,10))
  for i in range(n_pairs_diff):
  
    if train_or_test == "train" :
      classes = np.random.choice(range(0,5), size = 2, replace = False)

    elif train_or_test == "test" :
      classes = np.random.choice(range(0,10), size = 2, replace = False)
    else :
      raise NameError('Specify if train or train')
    
    classe_1 = classes[0]
    classe_2 = classes[1]
    
    row_ind1 = random.sample(range(all_data[classe_1].shape[0]), 1)
    indiv1 = all_data[classe_1][row_ind1]

    row_ind2 = random.sample(range(all_data[classe_2].shape[0]), 1)
    indiv2 = all_data[classe_2][row_ind2]
    
    labels_diff[i,0] = classe_1
    labels_diff[i,1] = classe_2

    pairs_diff[i,:,0] = indiv1
    pairs_diff[i,:,1] = indiv2

  return (pairs_diff,labels_diff)

def euclidean_distance(vects):
  x, y = vects
  sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
  shape1, shape2 = shapes
  return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
  margin = 1
  square_pred = K.square(y_pred)
  margin_square = K.square(K.maximum(margin - y_pred, 0))

  return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(predictions, labels,threshold):
  '''Compute classification accuracy with a fixed threshold on distances.
'''
  pred = predictions.ravel() < threshold
  return np.mean(pred == labels)

def compute_accuracy2(predictions, labels,threshold):
  '''Compute classification accuracy with a fixed threshold on distances.
'''
  pred = predictions.ravel() > threshold
  return np.mean(pred == labels)

# do PCA for the projected data
def fitting_pca(encodings,nb_components):
  pca = PCA(n_components=nb_components)
  pca.fit(encodings)
  projected = pca.transform(encodings)
  
  return pca,projected

def transformed_encodings_plot_2d(transformed_encodings,labels):
  
  df = pd.DataFrame(transformed_encodings[:, :2], columns=['Proj1', 'Proj2'])
  df['label'] = labels
  plt.figure(figsize=(8,8))
  ax = sns.scatterplot('Proj1', 'Proj2', data=df,
                  palette='tab10',
                  hue='label',
                  linewidth=0,
                  alpha=0.6
                )
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

def threshold_plot(pred_test, y_test,pred_train, y_train):
  threshold_list = [i*0.02 for i in range (1,100)]
  test_acc_list = [compute_accuracy(pred_test, y_test,threshold) for threshold in threshold_list]
  train_acc_list = [compute_accuracy(pred_train, y_train,threshold) for threshold in threshold_list]
  train_index_max = np.argmax(np.array(train_acc_list))
  test_index_max = np.argmax(np.array(test_acc_list))

  plt.figure(figsize = (8,8))
  plt.plot(threshold_list,train_acc_list,label='train accuracy as a function of threshold', c='green')
  plt.axvline(x=threshold_list[train_index_max], color='green',linestyle='--',label='best value of threshold for train = {:.2f}'.format(threshold_list[train_index_max]))
  plt.plot(threshold_list,test_acc_list,label='test accuracy as a function of threshold', c='red')
  plt.axvline(x=threshold_list[test_index_max], color='red',linestyle='--',label='best value of threshold for test = {:.2f}'.format(threshold_list[test_index_max]))
  plt.xlabel('threshold values')
  plt.ylabel('accuracy as a function of threshold')
  plt.title('influence of the threshold on the accuracy')
  plt.legend()
  plt.show()

  return threshold_list[train_index_max],threshold_list[test_index_max]

def threshold_plot2(pred_test, y_test,pred_train, y_train):

  threshold_list = [i*0.01 for i in range (1,100)]
  test_acc_list = [compute_accuracy2( pred_test, y_test,threshold) for threshold in threshold_list]
  train_acc_list = [compute_accuracy2(pred_train, y_train,threshold) for threshold in threshold_list]
  train_index_max = np.argmax(np.array(train_acc_list))
  test_index_max = np.argmax(np.array(test_acc_list))

  plt.figure(figsize = (8,8))
  plt.plot(threshold_list,train_acc_list,label='train accuracy as a function of threshold', c='green')
  plt.axvline(x=threshold_list[train_index_max], color='green',linestyle='--',label='best value of threshold for train = {:.2f}'.format(threshold_list[train_index_max]))
  plt.plot(threshold_list,test_acc_list,label='test accuracy as a function of threshold', c='red')
  plt.axvline(x=threshold_list[test_index_max], color='red',linestyle='--',label='best value of threshold for test = {:.2f}'.format(threshold_list[test_index_max]))
  plt.xlabel('threshold values')
  plt.ylabel('accuracy as a function of threshold')
  plt.title('influence of the threshold on the accuracy')
  plt.legend()
  plt.show()

  return threshold_list[train_index_max],threshold_list[test_index_max]


### prepare pairs of same class using all classes ############################

def prepare_same_pairs2(all_data,n_pairs_same,train_or_test):
  pairs_same = np.zeros((n_pairs_same,img_size,2))
  labels_same = np.zeros(n_pairs_same) ## keep track of the labels

  for i in range(n_pairs_same):
    ## for each time you want to create a pair of same class 
    ## first choose a class at random :  class  between 0 and 4 then randomly select two examples of that class from all_train[class]
    ## pairs_same[0,:,i] is the first example from the ith pair
    ## pairs_same[1,:,i] is the second example from the ith pair , they both are from same class
    if train_or_test == "train":
      classe = np.random.randint(0,5)
    elif train_or_test == "test" :
      classe = np.random.randint(0,10)
    else :
      raise NameError('Specify if train or train')
    row_ind1 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv1 = all_data[classe][row_ind1]
    row_ind2 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv2 = all_data[classe][row_ind2]
    label = classe
    labels_same[i] = label
    pairs_same[i,:,0] = indiv1
    pairs_same[i,:,1] = indiv2


  if train_or_test == "train":
    for i in range (5,10):
      pairs_same[n_pairs_same-10+i,:,0] = all_data[i][0]
      pairs_same[n_pairs_same-10+i,:,1] = all_data[i][0]
      labels_same[n_pairs_same-10+i] = i


  return (pairs_same,labels_same)


### prepare pairs of different class using all classes ############################

def prepare_different_pairs2(all_data, n_pairs_diff,train_or_test):

  pairs_diff = np.zeros((n_pairs_diff,img_size,2))
  labels_diff = np.zeros((n_pairs_diff,2)) ## keep track of the labels 

     ## for each time you want to create a pair of different class 
    ## first choose two different classes at random :  classes_1 and classes_2  between 0 and 4 then randomly select one example from each classe from all_train[classes_1]
    ## and from all_train[classes_2]
    ## pairs_diff[0,:,i] is the first example from the ith pair
    ## pairs_diff[1,:,i] is the second example from the ith pair , they both are from different classes
  classes = [-1,-1]
  list_of_range = list(range(0,10))
  for i in range(n_pairs_diff):

    if train_or_test == "train" :
      if i% 10 == 0:
        classes[0] = 5
      if i% 10 == 2:
        classes[0] = 6
      if i% 10 == 4:
        classes[0] = 7
      if i% 10 == 6:
        classes[0] = 8
      if i% 10 == 8:
        classes[0] = 9
      classes[1] = np.random.choice(range(0,5))
    elif train_or_test == "test" :
      classes = np.random.choice(range(0,10), size = 2, replace = False)
    else :
      raise NameError('Specify if train or train')
    
    classe_1 = classes[0]
    classe_2 = classes[1]
    
    row_ind1 = random.sample(range(all_data[classe_1].shape[0]), 1)
    indiv1 = all_data[classe_1][row_ind1]

    row_ind2 = random.sample(range(all_data[classe_2].shape[0]), 1)
    indiv2 = all_data[classe_2][row_ind2]
    
    labels_diff[i,0] = classe_1
    labels_diff[i,1] = classe_2

    pairs_diff[i,:,0] = indiv1
    pairs_diff[i,:,1] = indiv2
 
  if train_or_test == "train":
    k = 0
    for i in range(5,10):
      for j in range(i+1,10):
          labels_diff[n_pairs_diff-10+k,0] = i
          labels_diff[n_pairs_diff-10+k,1] = j

          pairs_diff[n_pairs_diff-10+k,:,0] = all_data[i][0]
          pairs_diff[n_pairs_diff-10+k,:,1] = all_data[j][0]
          k +=1

  
  return (pairs_diff,labels_diff)

def transformed_encodings_plot_3d(pca,transformed_encodings,labels):
  df = pd.DataFrame(transformed_encodings[:, :2], columns=['Proj1', 'Proj2'])
  df['label'] = labels
  total_var = pca.explained_variance_ratio_.sum() * 100

  fig = px.scatter_3d(
      transformed_encodings, x=0, y=1, z=2, color=df['label'],
      title=f'Total Explained Variance: {total_var:.2f}%',
      labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
  )
  fig.show()

def generate_triplet(x,y,testsize=0.3,ap_pairs=10,an_pairs=10):
  data_xy = tuple([x,y]) #transform as a tuple 
  trainsize = 1-testsize
  triplet_train_pairs = []
  triplet_test_pairs = []
  for data_class in sorted(set(data_xy[1])): # for element in labels   (sorted(set(data_xy[1])) returns [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    same_class_idx = np.where((data_xy[1] == data_class))[0]  # gather indexes of elements of the same class as data_class
    diff_class_idx = np.where(data_xy[1] != data_class)[0] #gather indexes of elements of different class to data_class
    A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs) #Generating Anchor-Positive pairs
    Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)#returns 10 idx of elements of different class than data_class
      #train
    A_P_len = len(A_P_pairs)
    Neg_len = len(Neg_idx)
    for ap in A_P_pairs[:int(A_P_len*trainsize)]:
      Anchor = data_xy[0][ap[0]]  #in list anchor gather all the x-train of anchor idx 
      Positive = data_xy[0][ap[1]] #in list anchor gather all the x-train of positive idx 
      for n in Neg_idx:
        Negative = data_xy[0][n]
        triplet_train_pairs.append([Anchor,Positive,Negative])               
      #test
    for ap in A_P_pairs[int(A_P_len*trainsize):]:
      Anchor = data_xy[0][ap[0]]
      Positive = data_xy[0][ap[1]]
      for n in Neg_idx:
        Negative = data_xy[0][n]
        triplet_test_pairs.append([Anchor,Positive,Negative])                    
  return (np.array(triplet_train_pairs), np.array(triplet_test_pairs))

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss


def create_base_network(input_shape):
  input = Input(shape=input_shape)
  mod = Sequential()
  mod.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  mod.add(MaxPooling2D(pool_size=(2, 2)))
  # mod.add(Conv2D(64, (3, 3), activation='relu'))
  # mod.add(MaxPooling2D(pool_size=(2, 2)))
  mod.add(Dropout(0.2))
  mod.add(Flatten())
  mod.add(Dense(3, activation='relu'))
  return mod

def n_way_one_shotlearning (base_net,all_train,all_test, k , typ,size_of_encoding,eary_or_hard):

  accuracy_on_k_tests = []

  for i in range(k): ## k is the number of times we repeat the process

    known_data = np.zeros((10,all_train[0].shape[1])) ## list of known images from 10 classes

    encodings_of_known_data = np.zeros((10,size_of_encoding)) ## list of their encodings

    if typ == 'contrastive' or typ =='binary_crossentropy': 
      for classe in range(10):

        ind = np.random.randint(0,all_train[classe].shape[0]) ## choose an example per class

        example = all_train[classe][ind,:]

        known_data[classe,:] = example ## add it to known data

        encodings_of_known_data[classe,:] = np.array(base_net.predict(example.reshape((-1,32,32,3))))
      
      ## choose a test class
      if eary_or_hard == 'easy':
        test_class = np.random.randint(0,5) 
      elif eary_or_hard == 'medium':
        test_class = np.random.randint(0,10) 
      elif eary_or_hard == 'hard':
        test_class = np.random.randint(5,10)
      else :
         raise NameError('Specify if easy or hard')

      test_index = np.random.randint(0,all_test[test_class].shape[0])

      test_image = all_test[test_class][test_index,:] ### choose a test image from the class

      test_encoding = np.array(base_net.predict(test_image.reshape((-1,32,32,3))))

    if typ == 'triplet': 

      for classe in range(10):
        ind = np.random.randint(0,all_train[classe].shape[0]) ## choose an example per class

        example = all_train[classe][ind,:]

        known_data[classe,:] = example## add it to known data

        encodings_of_known_data[classe,:] = np.array(base_net.predict(example.reshape((-1,32,32,3))))

      ## choose a test class
      if eary_or_hard == 'easy':
        test_class = np.random.randint(0,5) 
      elif eary_or_hard == 'medium':
        test_class = np.random.randint(0,10) 
      elif eary_or_hard == 'hard':
        test_class = np.random.randint(5,10)
      else :
         raise NameError('Specify if easy or hard')
         
      test_index = np.random.randint(0,all_test[test_class].shape[0])

      test_image = all_test[test_class][test_index,:] ### choose a test image from the class

      test_encoding = np.array(base_net.predict(test_image.reshape((-1,32,32,3))))

      # print('shape_of_encoding_of_test',test_encoding.shape)
    prediction = 0

    min_distance = 50000000000000000
    for j in range(10):
      distance = euclidean_distance((encodings_of_known_data[j],test_encoding ))
      # print(np.float(distance[0][0]))
      if distance < min_distance :
        prediction = j
        min_distance = distance
    if test_class == prediction :
      accuracy_on_k_tests+=[1]
    else :
      accuracy_on_k_tests+=[0]

  return(accuracy_on_k_tests, np.mean(accuracy_on_k_tests))
def prepare_same_pairs_test(all_data,n_pairs_same,easy_or_hard):
  pairs_same = np.zeros((n_pairs_same,3072,2))
  labels_same = np.zeros(n_pairs_same) ## keep track of the labels

  for i in range(n_pairs_same):
    ## for each time you want to create a pair of same class 
    ## first choose a class at random :  class  between 0 and 4 then randomly select two examples of that class from all_train[class]
    ## pairs_same[0,:,i] is the first example from the ith pair
    ## pairs_same[1,:,i] is the second example from the ith pair , they both are from same class
    if easy_or_hard == "easy":
      classe = np.random.randint(0,5)
    elif easy_or_hard == "medium":
      classe = np.random.randint(0,10)
    elif easy_or_hard == "hard" :
      classe = np.random.randint(5,10)
    else :
      raise NameError('Specify if easy or hard')
    row_ind1 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv1 = all_data[classe][row_ind1]
    row_ind2 = random.sample(range(all_data[classe].shape[0]), 1)
    indiv2 = all_data[classe][row_ind2]
    label = classe
    labels_same[i] = label
    pairs_same[i,:,0] = indiv1
    pairs_same[i,:,1] = indiv2

  return (pairs_same,labels_same)
def prepare_different_pairs_test(all_data, n_pairs_diff,easy_or_hard):

  pairs_diff = np.zeros((n_pairs_diff,3072,2))
  labels_diff = np.zeros((n_pairs_diff,2)) ## keep track of the labels 

    ## for each time you want to create a pair of different class 
    ## first choose two different classes at random :  classes_1 and classes_2  between 0 and 4 then randomly select one example from each classe from all_train[classes_1]
    ## and from all_train[classes_2]
    ## pairs_diff[0,:,i] is the first example from the ith pair
    ## pairs_diff[1,:,i] is the second example from the ith pair , they both are from different classes
  classes = [-1,-1]
  list_of_range = list(range(0,10))
  for i in range(n_pairs_diff):

    if easy_or_hard == "easy" :
      classes = np.random.choice(range(0,5), size = 2, replace = False)
    elif easy_or_hard == "medium" :
      classes[0] = np.random.randint(5,10)
      classes[1] = np.random.randint(0,10)
      while classes[0] == classes[1] :
        classes[1] = np.random.randint(0,10)
    elif easy_or_hard == "hard" :
      classes = np.random.choice(range(5,10), size = 2, replace = False)
    else :
      raise NameError('Specify if easy or hard')
    
    classe_1 = classes[0]
    classe_2 = classes[1]
    
    row_ind1 = random.sample(range(all_data[classe_1].shape[0]), 1)
    indiv1 = all_data[classe_1][row_ind1]

    row_ind2 = random.sample(range(all_data[classe_2].shape[0]), 1)
    indiv2 = all_data[classe_2][row_ind2]
    
    labels_diff[i,0] = classe_1
    labels_diff[i,1] = classe_2

    pairs_diff[i,:,0] = indiv1
    pairs_diff[i,:,1] = indiv2
  
  return (pairs_diff,labels_diff)



def loss_comparison_plot(contrastive_history,binary_history):
  x = None
  contras_train_loss = []
  contras_val_loss = []
  binary_train_loss = []
  binary_val_loss = []
  if len(contrastive_history.history['loss']) > len(binary_history.history['loss']) :
    x = range(1,len(contrastive_history.history['loss'])+1)
    contras_train_loss = contrastive_history.history['loss']
    contras_val_loss = contrastive_history.history['val_loss']
    binary_train_loss = binary_history.history['loss']+ [binary_history.history['loss'][-1]]*(len(contrastive_history.history['loss'])-len(binary_history.history['loss']))
    binary_val_loss = binary_history.history['val_loss']+ [binary_history.history['val_loss'][-1]]*(len(contrastive_history.history['loss'])-len(binary_history.history['loss']))
  elif len(contrastive_history.history['loss']) < len(binary_history.history['loss']) :
    x = range(1,len(binary_history.history['loss'])+1)
    contras_train_loss = contrastive_history.history['loss'] + [contrastive_history.history['loss'][-1]]*(-len(contrastive_history.history['loss'])+len(binary_history.history['loss']))
    contras_val_loss = contrastive_history.history['val_loss'] + [contrastive_history.history['val_loss'][-1]]*(-len(contrastive_history.history['loss'])+len(binary_history.history['loss']))
    binary_train_loss = binary_history.history['loss']
    binary_val_loss = binary_history.history['val_loss']
  else :
    x = range(1,len(contrastive_history.history['loss'])+1)
    contras_train_loss = contrastive_history.history['loss']
    contras_val_loss = contrastive_history.history['val_loss']
    binary_train_loss = binary_history.history['loss']
    binary_val_loss = binary_history.history['val_loss']
  plt.plot(x,contras_train_loss,'g',label='contrastive train')
  plt.plot(x,contras_val_loss,'g--',label='contrastive validation')
  plt.plot(x,binary_train_loss,'r',label='binary cross entropy train')
  plt.plot(x,binary_val_loss,'r--',label='binary cross entropy val')

#######
