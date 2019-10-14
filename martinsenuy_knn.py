# -*- coding: utf-8 -*-

######## Libraries ######## 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn import neighbors

#### EXAMPLE INITIAL DATA ####
######## Training data ########
n = 200 # Sample size
d_dimensions = 2 # Dimensions of the data

coords_1 = np.array(10 * np.random.normal(-2, 1.5, n)).reshape(-1, 2)
coords_df1 = pd.DataFrame(coords_1, columns = ['x-coord', 'y-coord'])
coords_df1['Class'] = 0
#coords_df1['Norm'] = 'Class 0' 

coords_2 = np.array(10 * np.random.normal(2, 2, n)).reshape(-1, 2)
coords_df2 = pd.DataFrame(coords_2, columns = ['x-coord', 'y-coord'])
coords_df2['Class'] = 1
#coords_df2['Norm'] = 'Class 1'

coords_df = coords_df1.append(coords_df2)


X = coords_df[['x-coord', 'y-coord']]
y = coords_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

################################################
########## Here Starts the function ############
# It will work with the example data or any data loaded in the right format
################################################
#### This code predicts the class (between 2 clasess) for 2-dimensional points
 
## X_train: Is the training data. The numeric attributes for each record. In
# this case the format should be DataFrame([:, 2])

## y_train: These are the training labels. i.e: 'Classes', 'State', etc.
# It can be in different formats with one column and the same rows as 'X_train'

## X_test: This variable receives the information of the record that we want
# to make the prediction on. Same format as 'X_train' but not necessary the
# same number of rows (records)

## y_test: This variable shows the actual label of the record to check our
# accuracy. Same format as 'y_train' and 'X_test''s number of records

## n_neighbors or K-parameter: Is the number of 'nearest' observations we
# think are necessary to predict the class of a record

def knn_classif(X_train, y_train, X_test, n_neighbors = 5):#, y_test):
    # If we have the labels and want to check the accuracy, we can include
    # the last parameter and the '#1' blocked lines 
    knn = KNeighborsClassifier(n_neighbors)    
    knn.fit(X_train, y_train)
    Train_accuracy = knn.score(X_train, y_train)
#1   Test_accuracy = knn.score(X_test, y_test)

    # The following 2 lines are just for the example, one has to remove it if
    # he loads all the variables
    X_test.reset_index(inplace = True) 
    X_test.drop(['index'], axis = 1, inplace = True)
    ###############################
    
    ck = []
        
    for i in range(len(X_test)):
       ck.append(knn.predict([X_test.loc[i,:]])[0])
 
    X_test['Predicted Class'] = ck
    print('')
    print('RESULTS')
    print('Train_accuracy: ' + str(Train_accuracy*100) + '%')
#1  print('Test_accuracy: ' + str(Test_accuracy*100) + '%')
    return X_test#, Train_accuracy, Test_accuracy
    

knn_classif(X_train, y_train, X_test, n_neighbors = 5) #, y_test)



########################################
#### Visualization of the classifier ###
y_test = X_test['Predicted Class']


def plot_class_knn(X_train, y_train, n_neighbors = 5, weights = 'uniform'):
    X_mat = X_train[['x-coord', 'y-coord']].values
    y_mat = y_train.values
# Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00'])
    cmap_bold2 = ListedColormap(['#0000FF', '#AFAFAF'])
                                 
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)
    
#############################################
#### Here I reached the hour of coding ######
#############################################

# Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .1  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
# Put the result into a color plot
    Z = Z.reshape(yy.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s = plot_symbol_size, c = y_train, 
                cmap = cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='Train 0')
    patch1 = mpatches.Patch(color='#00FF00', label='Train 1')
    #plt.legend(handles=[patch0, patch1], title = 'Training Classes')
    plt.xlabel('x-coords')
    plt.ylabel('y-coords')
    plt.title("2-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))    
    
    plt.scatter(X_test['x-coord'], X_test['y-coord'], c = y_test,
                cmap = cmap_bold2, edgecolor = 'black')
    patch2 = mpatches.Patch(color='#0000FF', label='Test 0')
    patch3 = mpatches.Patch(color='#AFAFAF', label='Test 1')
    plt.legend(handles=[patch0, patch1, patch2, patch3], title = 'Classes')
    
    #plt.legend(handles=[patch2, patch3], title = 'Test Classes')

    plt.ion()
    plt.show()
    plt.ioff()

plot_class_knn(X_train, y_train, 5, 'uniform')

########## Total coding time: 1 hr, 50 mins ##########