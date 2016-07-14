import time
import caffe
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from skimage.transform import resize
from scipy.sparse import csr_matrix

class FlowWarpingLayer(caffe.Layer):
    def create_A(self, disp):
        h = disp.shape[1]
        w = disp.shape[2]
        locs = np.indices(disp.shape[-2:]).astype(np.float)
        locs += disp

        nns = np.tile(np.floor(locs).astype(np.int), (4,1,1,1))
        nns[1][1] += 1
        nns[2][0] += 1
        nns[3] += 1
        
        coefs = np.zeros((w*h*4,))
        partial_i = np.zeros_like(coefs)
        partial_j = np.zeros_like(coefs)
        indices = np.zeros_like(coefs)

        
        stride4 = np.arange(0, w*h*4, 4);
        for i in range(4):
            is_valid = np.logical_and.reduce((nns[i][0] >= 0, nns[i][0] < h,
                nns[i][1] >= 0, nns[i][1] < w))
            nns[i,0][np.logical_not(is_valid)] = -1
            nns[i,1][np.logical_not(is_valid)] = -1
            indices[stride4 + i] = (nns[i][0] * w + nns[i][1]).ravel()
        
        rem = locs - nns[0]
        rem1 = 1 - rem

        coefs[stride4 + 0] = (rem1[0] * rem1[1]).ravel()
        coefs[stride4 + 1] = (rem1[0] * rem[1]).ravel()
        coefs[stride4 + 2] = (rem[0] * rem1[1]).ravel()
        coefs[stride4 + 3] = (rem[0] * rem[1]).ravel()
        ssum = coefs[stride4] + coefs[stride4 + 1] + coefs[stride4 + 2] + coefs[stride4 + 3] 
        #print '>'*10, ssum.max()
        partial_i[stride4 + 0] = (-rem1[1]).ravel()
        partial_i[stride4 + 1] = (-rem[1]).ravel()
        partial_i[stride4 + 2] = (rem1[1]).ravel()
        partial_i[stride4 + 3] = (rem[1]).ravel()

        partial_j[stride4 + 0] = (-rem1[0]).ravel()
        partial_j[stride4 + 1] = (rem1[0]).ravel()
        partial_j[stride4 + 2] = (-rem[0]).ravel()
        partial_j[stride4 + 3] = (rem[0]).ravel()
        
        #print '>'*10, (coefs.reshape((-1,4)).sum(1)).max()
        coefs = coefs[indices >= 0] 
        partial_i = partial_i[indices >= 0]
        partial_j = partial_j[indices >= 0]
        #ptrs = np.arange(0, h * w * 4 + 4, 4)
        num_valid = (indices.reshape((-1,4)) >= 0).sum(1)
        ptrs = np.concatenate(([0], np.cumsum(num_valid)))
        indices = indices[indices >= 0]

        coefs_mat = csr_matrix((coefs, indices, ptrs), shape=(w*h, w*h))
        partials_mats = [csr_matrix((partial_i, indices, ptrs), shape=(w*h, w*h)), 
                         csr_matrix((partial_j, indices, ptrs), shape=(w*h, w*h))]
        #print '>'*10, coefs_mat.sum(1).max()
        return (coefs_mat, partials_mats)


    def setup(self, bottom, top):
        pass   
      
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        n = bottom[0].data.shape[0]
        self.partials = [None] * n
        self.coefs = [None] * n
        for i in range(n):
            coefs, partials = self.create_A(bottom[1].data[i])
            top[0].data[i, ...] = (bottom[0].data[i].reshape((-1,
                coefs.shape[0])) * coefs.T).reshape(top[0].data[i].shape) 
            self.partials[i] = partials
            self.coefs[i] = coefs
            with open('coef%d.pkl'%i, 'wb') as f:
                pickle.dump(coefs, f, -1)
    
    def backward(self, top, propagate_down, bottom):
        n = bottom[0].data.shape[0]
        n_pix = self.partials[0][0].shape[0]
        for i in range(n):
            top_diff = top[0].diff[i]
            bottom[0].diff[i, ...] = (top_diff.reshape((-1, n_pix)) *
                self.coefs[i]).reshape(top_diff.shape)
            
            RoRd = bottom[0].data[i].reshape((-1, n_pix)) * self.partials[i][0].T
            ddiff = (RoRd * top_diff.reshape((-1, n_pix))).sum(0)
            print ddiff.shape, top_diff.shape[-2:]
            bottom[1].diff[i, 0, ...] = ddiff.reshape(top_diff.shape[-2:])
            
            bottom[1].diff[i, 1, ...] = ((bottom[0].data[i].reshape((-1, n_pix)) * self.partials[i][1].T) * top_diff.reshape((-1,n_pix))).sum(0).reshape(top_diff.shape[-2:])



