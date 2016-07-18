import json
import time
import caffe
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from transformer import Transformer_dist, Transformer
from skimage.transform import resize
from multiprocessing import Process, Queue
from util import  check_params, load_netflow_db, read_netflow_instance
import scipy as sp

class LoaderProcess(Process):
    def __init__(self, name=None, args=(),
                 kwargs=None):
        Process.__init__(self, name=name)
        self.queue = kwargs['queue']
        self.loader = kwargs['loader']

    def run(self):
        while True:
            item = self.loader.load_next_image()
            self.queue.put(item)

class NetflowTransformedDataLayerPrefetch(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['img1', 'img2','flow']
        params = eval(self.param_str)
        check_params(params, 
                batch_size=1, 
                split=None, 
                shuffle=True, 
                num_threads = 1, 
                max_queue_size = 1,
                annotation_file = '/mnt/sdc/FlyingChairs_release/FlyingChairs_train_val.txt',
                im_shape = None,
                mean1=np.array([0.411451, 0.432060, 0.450141]),
                mean2=np.array([0.431021, 0.410602, 0.448553]) )
        self.batch_size = params['batch_size']
        self.num_threads = params['num_threads']
        self.max_queue_size = params['max_queue_size']
        self.queue = Queue(self.max_queue_size)
        self.processes = [None]*self.num_threads
        self.netflow_db = load_netflow_db(params['annotation_file'], params['split'], shuffle=params['shuffle'])
        
        for i in range(self.num_threads):
            batch_loader = BatchLoader(params, self.netflow_db)
            self.processes[i] = LoaderProcess(kwargs={'queue':self.queue,'loader':batch_loader})
            self.processes[i].daemon = True
            self.processes[i].start()

        top[0].reshape(
                self.batch_size, 3, params['im_shape'][0],
                params['im_shape'][1]) #NOTE: current is devided by 2
        top[1].reshape(
                self.batch_size, 3, params['im_shape'][0],
                params['im_shape'][1])
        top[2].reshape(
                self.batch_size, 2, params['im_shape'][0],
                params['im_shape'][1])
        
    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            item = self.queue.get()

            top[0].data[itt,...] = item['img1'] #im1
            top[1].data[itt,...] = item['img2'] #im2
            top[2].data[itt,...] = item['flow'] #flow

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, params, netflow_db):
        self.batch_size = params['batch_size']
	self.netflow_db = netflow_db
        self.im_shape = params['im_shape']
        
        self.video_transformer = Transformer_dist({'transx_param':(0.0,0.0), 'transy_param':(0.0,0.0), 'rot_param':(0.0, 0.0), 
						   'zoomx_param':(1.0, 0.0), 'zoomy_param':(1.0, 0.0), 'shear_param':(0.0, 2)}, {'sigma_range':(.0, .01), 'gamma_range':(.95, 1.05),
						   'contrast_range':(.95, 1.05), 'brightness_sigma':.05, 'mult_rgb_range':(0.95, 1.05),
						   'blur_param':[(.92, 0), (.04, 2), (.03, 3), (.01, 5)]})

	sqrt2 = np.sqrt(2)
        self.frame_transformer = Transformer_dist({'transx_param':(0.0,0.27035499630719317/sqrt2), 'transy_param':(0.0,0.03967734490564475/sqrt2), 'rot_param':(0.0, 5), 
						   'zoomy_param':(1.0, 0.1625625379509229/sqrt2), 'zoomx_param':(1.0, 0.11389299050167503/sqrt2), 'shear_param':(0.0, 1)}, {'sigma_range':(.0, .01), 'gamma_range':(.99, 1.02),
						   'contrast_range':(.98, 1.02), 'brightness_sigma':.01, 'mult_rgb_range':(0.99, 1.01),
						   'blur_param':[(.92, 0), (.04, 2), (.03, 3), (.01, 5)]})
						   
        self.indexes = np.arange(netflow_db['length'])
	self.cur = netflow_db['length']
	self.shuffle = params['shuffle']
	self.mean1 = np.array(params['mean1']).reshape(1,1,3)
        self.mean2 = np.array(params['mean2']).reshape(1,1,3)

	
    def load_next_image(self):
        if self.cur == self.netflow_db['length']:
            self.cur = 0
            if self.shuffle:
		random.shuffle(self.indexes)
	
        img1, img2, flow = read_netflow_instance(self.netflow_db,
                self.indexes[self.cur])

        flow = flow.transpose((2,0,1))
        if img2.shape[:2] != self.im_shape:
            raise Exception
        
        frame1_tran = self.video_transformer.sample()
        
        frame2_tran = self.frame_transformer.sample()
        image1 = frame1_tran.transform_img(img1.copy(), img1.shape[:2]) 
        image2 = frame2_tran.transform_img(img2.copy(), img2.shape[:2])
        image1 -= self.mean1
        image2 -= self.mean2
        #TODO: transform flow and indices
        # final_flow[T1(i,j)] = T2( (i,j) + f1(i,j) ) - T1(i,j)
        # final_flow[ m, n ] = flow_trans(i,j)  i,j \in Z
        # T1(i,j) = (m,n)
        # final_flow[m,n] = flow_trans(T1^-1(m,n))


        # 1) I[i,j], T
        # 2) IT[k,l] = I[T^-1(k,l)]
        # 1 and 2 ==> flow_trans(T1^-1(m,n)) = flow_trans_T[m,n]


        #If we apply T1 on flow_trans we will get final_flow
        newx = np.arange(img1.shape[1])
        newy = np.arange(img1.shape[0])
        mesh_grid = np.meshgrid(newx, newy)
        locs1 = mesh_grid
        locs2 = locs1 + flow
        x,y = frame1_tran.transform_points(locs1[0].ravel(), locs1[1].ravel(), locs1[0].shape)
        locs1 = np.concatenate((x,y)).reshape(flow.shape)
        x,y = frame2_tran.transform_points(locs2[0].ravel(), locs2[1].ravel(), locs2[0].shape)
        locs2 = np.concatenate((x,y)).reshape(flow.shape)
        flow_trans = locs2 - locs1
        
        final_flow = np.zeros(flow.shape)
        frame1_tran.color_adjustment_param = None
        final_flow[0] = frame1_tran.transform_img(flow_trans[0], flow_trans[0].shape)
        final_flow[1] = frame1_tran.transform_img(flow_trans[1], flow_trans[1].shape)
        
        # Cropping and resizing
        self.cur += 1
        item = {'img1':image1.transpose((2,0,1)),
                'img2' :image2.transpose((2,0,1)),
                'flow'   : final_flow[::-1, :, :]
                }


        return item
