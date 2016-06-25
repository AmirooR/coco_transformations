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


class DavisDataLayerClient(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params)
        self.port = params['port']
        
	self.context = zmq.Context.instance()
	self.sock = self.context.socket(zmq.REQ)
	self.sock.bind('tcp://*:' + self.port)
        print bcolors.OKBLUE, 'client set up', bcolors.ENDC
        #self.sock.send_pyobj(np.zeros((256,256)) )
        #self.sock.recv()
        #print bcolors.OKGREEN + 'dummy message sent'+bcolors.ENDC
	

    def forward(self, bottom, top):
	#mask is a Nx2xHxW dimensional matrix
	fg_mask = bottom[0].data[:, 0,:,:] #TODO avaz
        print bcolors.WARNING + 'client sends '+ str(fg_mask.shape) + bcolors.ENDC 
	self.sock.send_pyobj(fg_mask)
        print bcolors.WARNING + 'client waiting for response' + bcolors.ENDC
	response = self.sock.recv()
        if response != 'OK':
           print bcolors.FAIL, 'FATAL ERROR: response is not OK', bcolors.ENDC

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass
def check_params(params):
    required = ['port']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
