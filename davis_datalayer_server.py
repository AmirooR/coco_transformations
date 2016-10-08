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
from util import cprint, bcolors, bbox, crop, crop_undo, check_params, read_davis_frame, load_davis_sequences, add_noise_to_mask
from operator import itemgetter 
from functools import partial
from skimage.transform import resize
import copy

import os.path as osp
import os
import multiprocessing
import multiprocessing.pool

def unwarp_batch_loader_load_frame(args,**kwargs):
    return SequenceLoader.load_frame(*args,**kwargs)

class Video:
    def __init__(self, seq_index = -1, frame_index = sys.maxint, lenght = 0):
	self.seq_index = seq_index
	self.frame_index = frame_index
	self.predicted_mask = None
	self.mask_crop_param = None
	self.name = None
	self.lenght = lenght
	self.step = 0
	
    def is_finished(self):
	return self.seq_index < 0 or self.frame_index >= (self.lenght-2)

class SequenceLoaderProcess(Process):
    def __init__(self, name=None, args=(), kwargs=None):
        Process.__init__(self, name=name)
        self.batch_size = kwargs['batch_size']
        self.port = kwargs['port']
        self.queue = kwargs['queue']
        self.loader = kwargs['loader']
	self.result_dir = kwargs['result_dir']
	self.save_result = (self.result_dir is not None and self.result_dir != '')
	self.videos = [Video() for i in xrange(self.batch_size)]
	self.next_seq_index = 0
	
	self.mutex = multiprocessing.Lock()
	#Initiate db
        self.firstTime = True
	
    def run(self):
	try:
	    if self.save_result:
		from skimage.io import imsave
	    
	    #Initiate socket
	    self.context = zmq.Context.instance()
	    self.sock = self.context.socket(zmq.REP)
	    self.sock.connect('tcp://localhost:' + self.port)
	    cprint ('Server started', bcolors.OKBLUE)
	    next_frame_reader = partial(SequenceLoaderProcess.load_next_frame, self)
	    while True:
		load_new_indices = [i for i in range(self.batch_size) if self.videos[i].is_finished()]
		load_next_indices = [i for i in range(self.batch_size) if not self.videos[i].is_finished()]
		#cprint ('load_new_indices lenght: ' + str(len(load_new_indices)) + ' load_next_indices lenght: ' + str(len(load_next_indices)), bcolors.WARNING)
		items = [None] * self.batch_size
		if self.save_result:
		    video_copy = copy.deepcopy(self.videos)
		    
		#1) Those frames that do not require previous prediction will be loaded immediately
		if len(load_new_indices) > 1:
		    pool = multiprocessing.pool.ThreadPool(len(load_new_indices))
		    new_items = pool.map(next_frame_reader, load_new_indices)
		    pool.close()
		    pool.join()
		    for i in range(len(new_items)):
			items[load_new_indices[i]] = new_items[i]
		else:
		    for i in load_new_indices:
			items[i] = next_frame_reader(i)
		    
		
		#2) Read the predicted masks and compute the rest
		#check if we have a message to read
		if not self.firstTime:
		    predicted_mask = self.read_message()
		    for i in load_next_indices:
			self.videos[i].predicted_mask = predicted_mask[i]
		
		if len(load_next_indices) > 1:
		    pool = multiprocessing.pool.ThreadPool(len(load_next_indices))
		    next_items = pool.map(next_frame_reader, load_next_indices)
		    pool.close()
		    pool.join()
		    for i in range(len(next_items)):
			items[load_next_indices[i]] = next_items[i]
		else:
		    for i in load_next_indices:
			items[i] = next_frame_reader(i)
			    
		
		for i in range(self.batch_size):
		    self.queue.put(items[i])
		
		if self.save_result:
		    for i in range(self.batch_size):
			#Save Segmentation Results
			if not self.firstTime:
			    video_dir = osp.join(self.result_dir, video_copy[i].name)
			    if video_copy[i].step != 1:
				video_dir += str(video_copy[i].step)
			    if not osp.exists(video_dir):
				os.makedirs(video_dir)
			    p_mask = crop_undo(predicted_mask[i], **video_copy[i].mask_crop_param)
			    p_mask[p_mask < .5] = 0
			    p_mask[p_mask > 0] = 1
			    #p_mask = predicted_mask[i]
			    imsave(osp.join(video_dir, '%05d.png' % (video_copy[i].frame_index + 1)), p_mask)
			#Save Initialization mask
			if self.videos[i].frame_index == 0:
			    video_dir = osp.join(self.result_dir, self.videos[i].name)
			    if self.videos[i].step != 1:
				video_dir += str(self.videos[i].step)
			    if not osp.exists(video_dir):
				os.makedirs(video_dir)
			    small_mask = items[i]['current_mask']
			    if self.loader.scale_256:
				small_mask = small_mask / 255.0 + self.loader.mask_mean
			    p_mask = crop_undo(small_mask, **self.videos[i].mask_crop_param)
			    p_mask[p_mask < .5] = 0
			    p_mask[p_mask > 0] = 1
			    #p_mask = small_mask
			    imsave(osp.join(video_dir, '%05d.png' % self.videos[i].frame_index), p_mask)
		self.firstTime = False
	except:
	    cprint ('An Error Happended in run()',bcolors.FAIL)
	    cprint (str("".join(traceback.format_exception(*sys.exc_info()))), bcolors.FAIL)
	    self.queue.put(None)
	    raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    
    def load_next_frame(self, i):
	val = None
	while val is None:
	    if self.videos[i].is_finished():
		self.mutex.acquire()
		self.update_video(i)
		self.mutex.release()
	    else:
		self.videos[i].frame_index += 1
	    try:
		val = self.loader.load_frame(self.videos[i].seq_index, self.videos[i].frame_index, self.videos[i].predicted_mask, self.videos[i].mask_crop_param)
	    except Exception as e:
		info = '\nwhen reading ' + self.videos[i].name + ' ' + str(self.videos[i].frame_index)
		raise type(e), type(e)(e.message + info), sys.exc_info()[2]

	    if val is None:
		self.videos[i].mask_crop_param = None
		self.videos[i].predicted_mask = None
	
	self.videos[i].predicted_mask = None
	self.videos[i].mask_crop_param = val['label_crop_param']
	return val
    
    def update_video(self, i):
	self.videos[i].seq_index = self.next_seq_index
	self.videos[i].lenght = self.loader.sequences[self.next_seq_index]['num_frames']
        self.videos[i].frame_index = 0
        self.videos[i].predicted_mask = None
	self.videos[i].mask_crop_param = None
	self.videos[i].name = self.loader.sequences[self.next_seq_index]['name']
	self.videos[i].step = self.loader.sequences[self.next_seq_index]['step']
        self.next_seq_index = (self.next_seq_index + 1) % len(self.loader.sequences)
        if self.videos[i].lenght < 2:
	    self.update_video(i)

    def read_message(self):
        cprint('server before receive', bcolors.WARNING)
	message = self.sock.recv_pyobj()
        cprint('server received message ' + str(message.shape), bcolors.WARNING)
        self.sock.send('OK')
       	return message

class SequenceLoader(object):
    def __init__(self, params):
        self.resizeShape1 = params['cur_shape']
        self.resizeShape2 = params['next_shape']
        self.bgr = params['bgr']
        self.flow_params = params['flow_params']
        self.flow_method = params['flow_method']
        self.scale_256 = params['scale_256']
        self.bb1_enlargment = params['bb1_enlargment']
        self.bb2_enlargment = params['bb2_enlargment']
        self.mask_threshold = params['mask_threshold']
        
        #Augmentation methods
        self.noisy_mask = ('noisy_mask' in params['augmentations'])
        self.sequences = load_davis_sequences(params['db_sets'], max_seq_len = params['max_len'], shuffle=params['shuffle'], reverse_seq=('reverse_video' in params['augmentations']))
        self.mask_mean = params['mask_mean']
        self.mean = np.array(params['mean']).reshape(1,1,3)
	assert (len(self.flow_params) == 0) or (self.resizeShape1 == self.resizeShape2 and self.bb1_enlargment == self.bb2_enlargment)
	if self.bgr:
	    #Always store mean in RGB format
	    self.mean = self.mean[:,:, ::-1]
    
    def load_frame(self, seq, frame, mask1_cropped, mask1_crop_param):
        cprint('FRAME = ' + str(frame), bcolors.WARNING)
        
        #reading first frame
        fresh_mask = True
        frame1_dict = read_davis_frame(self.sequences, seq, frame)
        image1 = frame1_dict['image']
        mask1 = frame1_dict['mask']
        if (mask1 > .5).sum() < 500:
		return None
        if mask1_cropped is not None and mask1_crop_param is not None:
	    #convert mask1 to its original shape using mask1_crop_param
	    uncrop_mask1 = crop_undo(mask1_cropped, **mask1_crop_param)
	    inter = np.logical_and((mask1 > .5), uncrop_mask1 > .5).sum()
	    union = np.logical_or((mask1 > .5), uncrop_mask1 > .5).sum()
	    
	    if float(inter)/union > .1:
		mask1 = uncrop_mask1
		fresh_mask = False
	
	#reading second frame
	frame2_dict = read_davis_frame(self.sequences, seq, frame + 1, self.flow_method)
	image2 = frame2_dict['image']
        mask2 = frame2_dict['mask']
	if not frame2_dict.has_key('iflow'):
	    frame2_dict['iflow'] = np.zeros((image2.shape[0], image2.shape[1], 2))
	
        # Cropping and resizing
        mask1[mask1 < .2] = 0
        mask1_bbox = bbox(mask1)
        cimg = crop(image1, mask1_bbox, bbox_enargement_factor = self.bb1_enlargment, output_shape = self.resizeShape1, resize_order = 3) - self.mean
        cmask = crop(mask1.astype('float32'), mask1_bbox, bbox_enargement_factor = self.bb1_enlargment, output_shape = self.resizeShape1)
        if self.noisy_mask and fresh_mask:
	    #print 'Adding Noise to the mask...'
	    cmask = add_noise_to_mask(cmask)
        cimg_masked = cimg * (cmask[:,:,np.newaxis] > self.mask_threshold)
        cimg_bg = cimg * (cmask[:,:,np.newaxis] <= self.mask_threshold)
        nimg = crop(image2, mask1_bbox, bbox_enargement_factor = self.bb2_enlargment, output_shape = self.resizeShape2, resize_order = 3) - self.mean
        label = crop(mask2.astype('float32'), mask1_bbox, bbox_enargement_factor = self.bb2_enlargment, output_shape = self.resizeShape2, resize_order = 0)
        label_crop_param = dict(bbox=mask1_bbox, bbox_enargement_factor=self.bb2_enlargment, output_shape=image1.shape[0:2])
	
	cmask -= self.mask_mean
	if self.bgr:
	    cimg = cimg[:,:, ::-1]
	    cimg_masked = cimg_masked[:, :, ::-1]
	    cimg_bg = cimg_bg[:, :, ::-1]
	    nimg = nimg[:, :, ::-1]
	    
	if self.scale_256:
	    cimg *= 255
	    cimg_masked *= 255
	    cimg_bg *= 255
	    nimg *= 255
	    cmask *= 255

	    
        item = {'current_image': cimg.transpose((2,0,1)),
                'fg_image' : cimg_masked.transpose((2,0,1)),
                'bg_image' : cimg_bg.transpose((2,0,1)),
                'current_mask' : cmask,
                'next_image'   :nimg.transpose((2,0,1)),
                'label'        : label,
                'label_crop_param' : label_crop_param}
	
	#crop inv_flow
	if len(self.flow_params) > 0:
	    inv_flow = frame2_dict['iflow']
	    max_val = np.abs(inv_flow).max()
	    if max_val != 0:
		inv_flow /= max_val
	    iflow = crop(inv_flow, mask1_bbox, bbox_enargement_factor = self.bb2_enlargment, resize_order=1, output_shape = self.resizeShape2, clip = False, constant_pad = 0)
	    
	    x_scale = float(iflow.shape[1])/(mask1_bbox[3] - mask1_bbox[2] + 1)/self.bb2_enlargment
	    y_scale = float(iflow.shape[0])/(mask1_bbox[1] - mask1_bbox[0] + 1)/self.bb2_enlargment
	    
	    for i in range(len(self.flow_params)):
		name, down_scale, offset, flow_scale = self.flow_params[i]
		pad = int(-offset + (down_scale - 1)/2)
		h = np.floor(float(iflow.shape[0] + 2 * pad) / down_scale)
		w = np.floor(float(iflow.shape[1] + 2 * pad) / down_scale)
		
		n_flow = np.pad(iflow, ((pad, int(h * down_scale - iflow.shape[0] - pad)), (pad, int(h * down_scale - iflow.shape[1] - pad)), (0,0)), 'constant')
		n_flow = resize( n_flow, (h,w), order = 1, mode = 'nearest', clip = False)
		n_flow[:, :, 0] *= max_val * flow_scale * x_scale / down_scale
		n_flow[:, :, 1] *= max_val * flow_scale * y_scale / down_scale
		
		n_flow = n_flow.transpose((2,0,1))[::-1, :, :]
		item[name] = n_flow
        return item

class DavisDataLayerServer(caffe.Layer): 
    def __del__(self):
	self.process.terminate()

    def setup(self, bottom, top):
        self.top_names = ['current_image', 'fg_image', 'bg_image', 'next_image','current_mask','label']
        self.flow_names = []
        params = eval(self.param_str)
        check_params(params, result_dir='', batch_size=1, split=None, port=None, im_shape=0, shuffle=False, max_len=0, bgr=False, scale_256=False, bb1_enlargment=2.2, bb2_enlargment=2.2, cur_shape = 0, next_shape = 0, mask_mean = .5, mean=None, flow_params = [], flow_method = 'None', mask_threshold = 0.5, augmentations = [])
        
        #For backward compatibility
        if params['next_shape'] == 0 or params['cur_shape'] == 0:
	    if params['im_shape'] == 0:
		raise Exception
	    params['next_shape'] = params['im_shape']
	    params['cur_shape'] = [params['im_shape'][0]/2, params['im_shape'][1]/2]
	    
 
	if params['split'] == 'training':
	    db_sets = ['training', 'test_pascal', 'training_pascal', 'training_segtrackv2', 'training_jumpcut']
	elif params['split'] == 'test':
	    db_sets = ['test']
	params['db_sets'] = db_sets
	
        self.batch_size = params['batch_size']
        self.queue = Queue(self.batch_size)
        self.process = SequenceLoaderProcess(kwargs={'queue':self.queue,'loader':SequenceLoader(params),'batch_size':self.batch_size, 'port':params['port'], 'result_dir':params['result_dir']})
        self.process.daemon = True
        self.process.start()

        top[0].reshape(self.batch_size, 3, params['cur_shape'][0],params['cur_shape'][1])
        top[1].reshape(self.batch_size, 3, params['cur_shape'][0], params['cur_shape'][1])
	top[2].reshape(self.batch_size, 3, params['cur_shape'][0], params['cur_shape'][1])
        top[3].reshape(self.batch_size, 3, params['next_shape'][0], params['next_shape'][1])
        top[4].reshape(self.batch_size, 1, params['cur_shape'][0], params['cur_shape'][1])
        top[5].reshape(self.batch_size, 1, params['next_shape'][0], params['next_shape'][1])
	
	for i in range(len(params['flow_params'])):
	    ##in out network we have flow_coordinate = down_scale * 'name'_coordinate + offset
	    ##in python resize we have flow_coordinate = down_scale * 'name'_coordinate + (down_scale - 1)/2 - pad
	    ## ==> (down_scale - 1)/2 - pad = offset ==> pad = -offset + (down_scale - 1)/2
	    
	    name, down_scale, offset, flow_scale = params['flow_params'][i]
	    pad = -offset + (down_scale - 1)/2
	    assert pad == int(pad) and pad >= 0 and offset <= 0
	    h = int(np.floor(float(params['next_shape'][0] + 2 * pad) / down_scale))
	    w = int(np.floor(float(params['next_shape'][1] + 2 * pad) / down_scale))
	    top[i + 6].reshape(self.batch_size, 2, h, w)
	    self.top_names.append(name)
	    self.flow_names.append(name)
	    
    def forward(self, bottom, top):
	cprint ('Queue size ' + str(self.queue.qsize()), bcolors.OKBLUE)
        for itt in range(self.batch_size):
            #im1, im1_masked, im2, label = self.batch_loader.load_next_image()
            item = self.queue.get()
	    if item is None:
		self.process.terminate()
		raise Exception
	 
            top[0].data[itt,...] = item['current_image'] #im1
            top[1].data[itt,...] = item['fg_image'] #im1_fg
	    top[2].data[itt,...] = item['bg_image'] #im1_bg
            top[3].data[itt,...] = item['next_image'] #im2
            top[4].data[itt,...] = item['current_mask'] #label
            top[5].data[itt,...] = item['label'] #label
	    for i in range(len(self.flow_names)):
		flow_name = self.flow_names[i]
		top[i + 6].data[itt,...] = item[flow_name] #inverse flow
		
    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass
