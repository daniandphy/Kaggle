#from sklearn import svm
import re
import numpy as np
import glob
import os
from PIL import Image
from sklearn import (decomposition,lda)
import skimage.io as io
import time
import csv
from collections import Counter
from scipy.stats import mode
from skimage.feature import (corner_harris,corner_peaks,daisy)
from scipy import linalg
from scipy.ndimage import filters
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

class ZCA(BaseEstimator, TransformerMixin):
 
    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy
 
    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self
 
    def transform(self, X):
        X = array2d(X)
        X_mean=np.mean(X, axis=0)
        X_transformed = X - X_mean
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed


class ListTools():
  #####following are tools for sorting a list 
	def tryint(self,s):
		try:
		   return int(s)
		except:
		   return s
	def alphanumkey(self,s):
		return[ self.tryint(c) for c in re.split('([0-9]+)',s)]
	def sort_nicely(self,l):
		return sorted(l,key=self.alphanumkey)


class ImgTransformTools(ListTools):
	def __init__(self,path,regex_name='*.png',num_pixels=3072):
		self.path=path
		self.regex_name=regex_name
		self.num_pixels=num_pixels
		

	def read_imgs_to_arr(self,**kwargs):
		self.flatten=True;self.verbose=0;self.graying=False;self.normalizing=False;self.gauss_filter=False
		for key,value in kwargs.iteritems():
		  setattr(self,key,value)
		#if self.flatten:
		file_open = lambda x,y: self.sort_nicely(glob.glob(os.path.join(x,y)))
		image_list=file_open(self.path,self.regex_name)
		if self.verbose:
		  print(image_list[0:20])
		image_set=[self.img_to_arr(image,graying=self.graying,normalizing=self.normalizing,\
		  flatten=self.flatten,gauss_filter=self.gauss_filter) \
		  for image in image_list ]
		return np.array(image_set)
		#else:
		 # image_set=io.ImageCollection(self.path+self.regex_name)
		  #return io.concatenate_images(image_set)
	
	
	def img_to_arr(self, image_path,graying=False,normalizing=False,flatten=True,gauss_filter=True):
		#image=np.array(Image.open(image_path))
		image=io.imread(image_path,as_grey=graying)
		if gauss_filter:
		  image=self.gaussian_blur_filter(image)
		if flatten:
		  img_shape=image.shape
		  image=image.reshape(np.product(img_shape))
		if graying and normalizing:
		  image=self.normalize(image)
		return image
	      
	def arr_to_vec(self,arr):
		vec=arr.reshape(1,self.num_pixels)
		return vec[0]
	def normalize(self,arr):
		amin=arr.min()
		amax=arr.max()
		rng=amax-amin
		return (arr-amin)*255/rng
	
	def gaussian_blur_filter(self,arr,sigma=0.5,filter_size=5.0,order=0):
		return filters.gaussian_filter(arr,sigma,order=order,truncate=filter_size)
	        
############################
class ArrayIOTools():
	

	def write_arr_to_file(self,data,file_name):
		out_file=open(file_name,'w')
		np.save(out_file,data)
		
	def write_arr_to_csv(self,data,file_name):
	        with open(file_name,'w') as csv_file:
                    writer=csv.writer(csv_file,delimiter=',')
                    i=0
                    for line in data:
		       i+=1
		       temp=[i]
		       temp.append(line)
		       writer.writerow(temp)
		
		
	def read_file_to_arr(self,file_name):
		inp_file=open(file_name,'r')
		return np.load(inp_file)
	      
	def read_csv_to_arr(self,inp_file,skip_header=0):
		return np.genfromtxt(inp_file,delimiter=',',skip_header=skip_header,dtype=None)
	      
######################################
class ImgClassificationTools(ArrayIOTools):
  
  
	def ldafit(self,X,y,n_components=10,verbose=False):
		self.reduction_method=lda.LDA(n_components=n_components)
		self.reduction_method.fit(X,y)
		
	def pcafit(self,X,n_components=10,verbose=False):
		#self.pca=decomposition.PCA(n_components=n_components,whiten=True)
		#self.reduction_method=decomposition
		#self.reduction_method=decomposition.RandomizedPCA(n_components=n_components,whiten=True)
		#self.reduction_method=decomposition.FactorAnalysis(n_components=n_components)
		##self.reduction_method=decomposition.KernelPCA(n_components=n_components,kernel="sigmoid") ###segmentation fault!!!
		self.reduction_method=decomposition.FastICA(n_components=n_components,whiten=True)
		self.reduction_method.fit(X)
		if verbose:
		  print ("explained variance ration %.2f" %self.pca.explained_variance_ratio_)
		  print("sum of explained variance ratio %.2f for %i components" %(np.cumsum(self.pca.explained_variance_ratio_),n_components))
		  
	def blockvoting(self,method,X_train,y_train,X_test,blocks=1,verbose=0,dump=False):
		if verbose:
		  st=time.time()
		pca_trans_fit=[]
		test_size=len(X_test)
		train_size=len(y_train)
		block_size=train_size // blocks
		all_results=np.zeros((test_size,blocks),dtype='|S10')
		
		for block in range(blocks):
		  start_ind=block*block_size
		  stop_ind=(block+1)*block_size
		  block_X_train,block_y_train=X_train[start_ind:stop_ind,:],y_train[start_ind:stop_ind]
		  
		  self.pcafit(block_X_train)
		  #self.ldafit(block_X_train,block_y_train,n_components=10,verbose=False)
		  reduced_block_X_train=self.reduction_method.transform(block_X_train)
		  reduced_X_test=self.reduction_method.transform(X_test)
		  if (verbose):
		    print("block: ",block,", reduced_block_X_train.shape: ",reduced_block_X_train.shape,", reduced_X_test.shape:",reduced_X_test.shape)
		  method.fit(reduced_block_X_train,block_y_train)
		  all_results[:,block]=method.predict(reduced_X_test)
		if verbose:
		  print(all_results[:10,:],"all_results[:10,:]")
		  print ("VOTE KNN: Number of mislabeld point out of a total %d point: %d"%(y_train.shape[0],(y_train[:,np.newaxis]!=all_results).sum()))
		if dump:
		  self.write_arr_to_csv(all_results,"ensemble.csv")
		if verbose:
		  et=time.time()
		  print ("total time is {} sec" .format(et-st))
		return np.array(mode(all_results,axis=1)[0])
	      
	      
	def make_features_daisy(self,gray_imgs,**kwargs):
		#for key,value in kwargs.iteritems():
		  #setattr(self,key,value)
		all_features=[(daisy(gray_img,**kwargs)).reshape(-1).astype(np.float32) for gray_img in gray_imgs]
		return np.array(all_features)
	
	def map_label_to_num(self,labels):
		labels_to_number_dict={label:i for i,label in enumerate(set(labels))}
		numbers=[labels_to_number_dict[label] for label in labels]
		return labels_to_number_dict,np.array(numbers)
	
	def map_num_to_label(self,my_nums,my_dict):
		inv_dict={v:k for k,v in my_dict.items()}
		print("inv_dict: ",inv_dict)
		labels=[inv_dict[i] for i in my_nums]
		return labels
		  
if __name__=="__main__":
  pass