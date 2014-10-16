import imageclassification as ic
import numpy as np
from sklearn.cluster import KMeans
###
from sklearn.feature_extraction import image
from skimage.color import rgb2gray
from skimage.feature import (corner_harris,corner_peaks,daisy)
#import cv2
####
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
from sklearn import (manifold, decomposition, ensemble, lda,random_projection,qda)
from sklearn.neighbors import KNeighborsClassifier
aiot=ic.ArrayIOTools()
import time


X_train_data=aiot.read_file_to_arr('X_train_data_unflatten' )
y_train_data=aiot.read_csv_to_arr('trainLabels.csv',skip_header=1)['f1']
X_test_data=aiot.read_file_to_arr('X_test_data_unflatten' )
ict=ic.ImgClassificationTools()



start_making_words=time.time()

X_tr,y_tr,X_test=X_train_data,y_train_data,X_test

############################make data
all_features=aiot.read_file_to_arr('../../BOW_daisy_features_normalized' )
print("all_features.shape",all_features.shape)
n_clusters=200
km=KMeans(n_clusters=n_clusters,n_jobs=-1)
km.fit(all_features)
print("cluster_centers_.shape:",km.cluster_centers_.shape) 
print("labels_:",km.labels_[:10],km.labels_.shape) 


all_words=[]
for i in range(len(X_test)):
  color_img=X_test[i,:,:,:]
  gray_img=rgb2gray(color_img)
  gray_img=normalize(gray_img)
  features=daisy(gray_img,step=2,radius=8)
  words=km.predict(features.reshape(-1,200))
  #print(words)
  words,bin_edges=np.histogram(words,bins=range(n_clusters))
  #print('words.shape',words.shape)
  all_words.append(words)
  
all_words=np.vstack(all_words)
print('all_words.shape: ',all_words.shape)
print('all_words[:2,:] : ',all_words[:2,:])


aiot.write_arr_to_file(all_words,'test_BOW_daisy_normalized_kmean200')


reduced_X_tr=aiot.read_file_to_arr('../../BOW_daisy_normalized_kmean200' )
reduced_X_test=aiot.read_file_to_arr('test_BOW_daisy_normalized_kmean200' )
reduced_X_tr,y_tr,reduced_X_test,y_test=reduced_X_tr,y_tr,reduced_X_tr
print ('reading done!!')
method='normalized__kmean200_SVC_rbf_gamma'
parameter=
print("parameter",parameter)
#clf=SVC(kernel='poly',gamma=parameter,degree=2)
clf=SVC(gamma=parameter)
#clf=LinearSVC()
#clf=qda.QDA(reg_param=parameter)

#clf=KNeighborsClassifier(n_neighbors=parameter)
  
clf.fit(reduced_X_tr,y_tr)
print('fitting done!!')
y_tr_pred=clf.predict(reduced_X_tr)
y_test_pred=clf.predict(reduced_X_test)

train_cm=confusion_matrix(y_tr,y_tr_pred)
  
train_misclassifications=(len(y_tr)-(np.sum(train_cm.diagonal())))/float(len(y_tr))
  
print("parameter:", parameter," train_misclassifications:",train_misclassifications)
train_cm =train_cm/train_cm.astype(np.float).sum(axis=1)

path_name_train='train_'+str(parameter)+'_'+method
  #print("path_name_train",path_name_train)
plot_confusion_mat(train_cm, name=path_name_train)
aiot.write_arr_to_csv(y_tr_pred, path_name_train+"_pred.csv")
  
path_name_test='test_'+str(parameter)+'_'+method  
aiot.write_arr_to_csv(y_test_pred, path_name_test+"_pred.csv")


end_making_words=time.time()
print('total time:  ', end_making_words - start_making_words)