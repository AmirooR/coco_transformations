import json
import time
import caffe
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from transformer import Transformer_dist
from skimage.transform import resize
from multiprocessing import Process, Queue
from davis import cfg
from davis.dataset.utils import db_read_info
import os.path as osp
import zmq
from util import bcolors, cprint, check_params

class DavisDataLayerClient(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params, port=None)
        self.port = params['port']
        
	self.context = zmq.Context.instance()
	self.sock = self.context.socket(zmq.REQ)
	self.sock.bind('tcp://*:' + self.port)
        cprint('client set up', bcolors.OKBLUE) 
        #self.sock.send_pyobj(np.zeros((256,256)) )
        #self.sock.recv()
        #print bcolors.OKGREEN + 'dummy message sent'+bcolors.ENDC
        if len(top) > 0:
	    top[0].reshape(1)

    def forward(self, bottom, top):
	#mask is a Nx2xHxW dimensional matrix
	exp = np.exp(bottom[0].data)
	fg_mask = exp[:, 1,:,:] / exp.sum(1)
	fg_mask = fg_mask > .5
	#fg_mask = bottom[0].data[:,0,:,:]
        cprint('client sends '+ str(fg_mask.shape), bcolors.WARNING)
	self.sock.send_pyobj(fg_mask)
        cprint('client waiting for response', bcolors.WARNING)
	response = self.sock.recv()
        if response != 'OK':
           cprint('FATAL ERROR: response is not OK', bcolors.FAIL)

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass
