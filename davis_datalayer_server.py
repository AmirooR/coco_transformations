import json
import time
import caffe
import numpy as np
import cPickle as pickle
import random
from multiprocessing import Process, Queue, Pool
from davis import cfg
import zmq
import sys
import traceback
import itertools
from util import cprint, bcolors, bbox, crop, crop_undo, check_params, read_davis_frame, load_davis_sequences
import multiprocessing

def unwarp_batch_loader_load_frame(args,**kwargs):
    return BatchLoader.load_frame(*args,**kwargs)

class LoaderProcess(Process):
    def __init__(self, name=None, args=(), kwargs=None):
        Process.__init__(self, name=name)
        self.batch_size = kwargs['batch_size']
        self.port = kwargs['port']
        self.queue = kwargs['queue']
        self.loader = kwargs['loader']
	self.prediction_shape = tuple(kwargs['prediction_shape'])
	self.cur_frame_indices = [0] * self.batch_size
	self.cur_seq_indices = [-1] * self.batch_size
	self.next_seq_index = 0
	
	#Initiate db
	self.sequences = kwargs['sequences']
        self.firstTime = True
 
	
    def run(self):
	try:
	    #Initiate socket
	    self.context = zmq.Context.instance()
	    self.sock = self.context.socket(zmq.REP)
	    self.sock.connect('tcp://localhost:' + self.port)
	    cprint ('Server started', bcolors.OKBLUE)

	    mask_predicted = np.zeros((self.batch_size, ) + self.prediction_shape)
	    mask_crop_param = [None] * self.batch_size
	    while True:
		#check if we have a message to read
		if not self.firstTime: #np.any(self.cur_frame_indices != 0):
		    mask_predicted = self.read_message()
			
		for i in xrange(self.batch_size):
		    if mask_predicted[i].sum() < 2000 and self.cur_frame_indices[i] != 0:
			self.cur_frame_indices[i] = sys.maxint
			cprint('OBJECT DISAPEARED! SEQUENCE WILL BE CHANGED', bcolors.WARNING)
		    self.update_indices(i)
		
		if self.batch_size > 1:
		    pool = multiprocessing.pool.ThreadPool(self.batch_size)
		    items = pool.map(unwarp_batch_loader_load_frame, zip([self.loader]*self.batch_size, self.cur_seq_indices, self.cur_frame_indices, mask_predicted, mask_crop_param))
		    pool.close()
		    pool.join()
		for i in range(self.batch_size):
		    if self.batch_size > 1:
			item = items[i]
		    else:
			item = self.loader.load_frame(self.cur_seq_indices[i], self.cur_frame_indices[i], mask_predicted[i], mask_crop_param[i])
		    mask_crop_param[i] = item['label_crop_param']
		    self.queue.put(item)
		self.firstTime = False
        except:
	    cprint ('An Error Happended in run()',bcolors.FAIL)
	    cprint (str("".join(traceback.format_exception(*sys.exc_info()))), bcolors.FAIL)
	    raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    def update_indices(self, i):
        if self.cur_seq_indices[i] < 0 or self.cur_frame_indices[i] >= (self.sequences[self.cur_seq_indices[i]]['num_frames']-2):
            self.cur_seq_indices[i] = self.next_seq_index
            self.next_seq_index = (self.next_seq_index + 1) % len(self.sequences)
            self.cur_frame_indices[i] = 0
            if self.sequences[self.cur_seq_indices[i]]['num_frames'] < 2:
		self.update_indices(i)
        else:
            self.cur_frame_indices[i] += 1

    def read_message(self):
        cprint('server before receive', bcolors.WARNING)
	message = self.sock.recv_pyobj()
        cprint('server received message ' + str(message.shape), bcolors.WARNING)
        self.sock.send('OK')
       	return message
		
class DavisDataLayerServer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['current_image', 'masked_image','next_image','label']
        params = eval(self.param_str)
        check_params(params, batch_size=1, split=None, port=None, im_shape=None, shuffle=False, max_len = 0)
        self.batch_size = params['batch_size']
        self.port = params['port']
        self.queue = Queue(self.batch_size)
        self.sequences = load_davis_sequences(params['split'], max_seq_len = params['max_len'], shuffle=params['shuffle'])
        batch_loader = BatchLoader(self.sequences, params)
        self.process = LoaderProcess(kwargs={'queue':self.queue,'loader':batch_loader,'batch_size':self.batch_size,'prediction_shape':params['im_shape'],'port':self.port,'sequences':self.sequences})
        self.process.daemon = True
        self.process.start()

        top[0].reshape(
                self.batch_size, 3, params['im_shape'][0]/2,
                params['im_shape'][1]/2)
        top[1].reshape(
                self.batch_size, 3, params['im_shape'][0]/2,
                params['im_shape'][1]/2)
        top[2].reshape(
                self.batch_size, 3, params['im_shape'][0],
                params['im_shape'][1])
        top[3].reshape(
                self.batch_size, 1, params['im_shape'][0],
                params['im_shape'][1])

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            #im1, im1_masked, im2, label = self.batch_loader.load_next_image()
            item = self.queue.get()

            top[0].data[itt,...] = item['current_image'] #im1
            top[1].data[itt,...] = item['current_masked'] #im1_masked
            top[2].data[itt,...] = item['next_image'] #im2
            top[3].data[itt,...] = item['label'] #label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, sequences, params):
        self.batch_size = params['batch_size']
        self.resizeShape1 = (params['im_shape'][0]/2, params['im_shape'][1]/2)
        self.resizeShape2 = params['im_shape']
        self.sequences = sequences
    
    def load_frame(self, seq, frame, mask1_cropped, mask1_crop_param, img1_bb_enlargement = 1.1, img2_bb_enlargement = 2.2):
        cprint('FRAME = ' + str(frame), bcolors.WARNING)
        
        #reading first frame
        if frame == 0:
            image1, mask1 = read_davis_frame(self.sequences, seq, frame)
        else:
	    #convert mask1 to its original shape using mask1_crop_param
	    image1 = read_davis_frame(self.sequences, seq, frame, False)
	    mask1 = crop_undo(mask1_cropped, **mask1_crop_param)
	
	#reading second frame
	image2, mask2 = read_davis_frame(self.sequences, seq, frame + 1)
        
        
        # Cropping and resizing
        mask1_bbox = bbox(mask1)
        cimg = crop(image1, mask1_bbox, bbox_enargement_factor = img1_bb_enlargement, output_shape = self.resizeShape1, resize_order = 3)
        cmask = crop(mask1.astype('float32'), mask1_bbox, bbox_enargement_factor = img1_bb_enlargement, output_shape = self.resizeShape1)
        cimg_masked = cimg * cmask[:,:,np.newaxis]
        nimg = crop(image2, mask1_bbox, bbox_enargement_factor = img2_bb_enlargement, output_shape = self.resizeShape2, resize_order = 3)
        label = crop(mask2.astype('float32'), mask1_bbox, bbox_enargement_factor = img2_bb_enlargement, output_shape = self.resizeShape2, resize_order = 0)
        label_crop_param = dict(bbox=mask1_bbox, bbox_enargement_factor=img2_bb_enlargement, output_shape=image1.shape[0:2])
	
        item = {'current_image': cimg.transpose((2,0,1)),
                'current_masked' : cimg_masked.transpose((2,0,1)),
                'current_mask' : cmask,
                'next_image'   :nimg.transpose((2,0,1)),
                'label'        : label,
                'label_crop_param' : label_crop_param}
        return item
