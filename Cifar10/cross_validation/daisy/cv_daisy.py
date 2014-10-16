import sys
##adding imageclassification to the python path
sys.path.append('/home/danial/Dropbox/kaggle/cifar_10')

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
from sklearn import grid_search

aiot=ic.ArrayIOTools()
import time

ict=ic.ImgClassificationTools()
start_making_words=time.time()
y_train_data=aiot.read_csv_to_arr('../../trainLabels.csv',skip_header=1)['f1']
reduced_X_tr=aiot.read_file_to_arr('../../reduced_X_daisy_train_data_unflatten_grayed' )

print ('reading done!!')
reduced_X_tr,y_tr=reduced_X_tr,y_train_data
print ('reading done!!')

method='normalized_SVC_rbf_gamma_C'
parameters={'kernel':['rbf'],'C':[7.5,10,12.5],'gamma':np.arange(200,375,25)}
base_method=SVC()


clf=grid_search.GridSearchCV(base_method,parameters,n_jobs=4)
clf.fit(reduced_X_tr,y_tr)
print("")
for score in clf.grid_scores_:
  print("scores: ", score)

print(method,"#best parameters:", clf.best_params_)
#################################
reduced_X_test=aiot.read_file_to_arr('../../reduced_X_daisy_test_data_unflatten_grayed' )
print("predicting")
y_test_pred=clf.predict(reduced_X_test)
aiot.write_arr_to_csv(y_test_pred, "best_test_pred.csv")
end_making_words=time.time()
print('#total making words time:  ', end_making_words - start_making_words)