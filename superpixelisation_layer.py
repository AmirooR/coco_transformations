import time
import caffe
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from skimage.transform import resize
from skimage import data, segmentation, color
from skimage.measure import regionprops
from scipy.sparse import csr_matrix
from util import  check_params

class SuperpixelisationLayer(caffe.Layer):
    def setup(self, bottom, top):
	params = eval(self.param_str)
	check_params(params, n_segs=400, compactness=30)
	self.n_segs = params['n_segs']
        self.compactness = params['compactness']
      
    def reshape(self, bottom, top):
        # top[0]: avg prediction N*1*W*H
        top[0].reshape(*bottom[1].data.shape)
    
    def forward(self, bottom, top):
        # bottom[0]: images N*3*W*H
        # bottom[1]: prediction N*1*W*H
        n = bottom[0].data.shape[0]
        for i in range(n):
            labels = segmentation.slic( bottom[0].data[i].transpose((1,2,0)), 
                    compactness=self.compactness, n_segments=self.n_segs)
            top[0].data[i, ...] = color.label2rgb(labels, bottom[1].data[i].transpose((1,2,0)), kind='avg').transpose((2,0,1)) #.reshape(top[0].data[i].shape)
    
    def backward(self, top, propagate_down, bottom):
        pass
