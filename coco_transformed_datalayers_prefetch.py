import json
import time
import caffe
from pycocotools.coco import COCO
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from pycocotools import mask
from transformer import Transformer_dist
from skimage.transform import resize
from multiprocessing import Process, Queue
from util import load_coco_db, read_coco_instance, bbox, crop, check_params,  add_noise_to_mask


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

class CocoTransformedDataLayerPrefetch(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['current_image', 'masked_image','next_image','label']
        params = eval(self.param_str)
        check_params(params, batch_size=1, split=None, im_shape=0, shuffle=True, num_threads = 1, max_queue_size = 1, data_dir = '/home/amir/coco',
		     cats = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
			     'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 
			     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
			     'giraffe', 'kite'], 
			     areaRng = [1500. , np.inf], iscrowd=False, mean=None, noisy_mask = False, bgr=False, scale_256=False, cur_shape=0, next_shape=0)
		     
	 #For backward compatibility
        if params['next_shape'] == 0 or params['cur_shape'] == 0:
	    if params['im_shape'] == 0:
		raise Exception('Either im_shape or (cur_shape, next_shape) parameters should be set')
	    params['next_shape'] = params['im_shape']
	    params['cur_shape'] = [params['im_shape'][0]/2, params['im_shape'][1]/2]
	
	
        self.batch_size = params['batch_size']
        self.num_threads = params['num_threads']
        self.max_queue_size = params['max_queue_size']
        self.queue = Queue(self.max_queue_size)
        self.processes = [None]*self.num_threads
        self.coco_dbs = load_coco_db(params['data_dir'], params['split'], cats=params['cats'], areaRng=params['areaRng'], iscrowd=params['iscrowd'], shuffle=params['shuffle'], chunck_num = self.num_threads)
        
        for i in range(self.num_threads):
            batch_loader = BatchLoader(params, self.coco_dbs[i])
            self.processes[i] = LoaderProcess(kwargs={'queue':self.queue,'loader':batch_loader})
            self.processes[i].daemon = True
            self.processes[i].start()
        #self.batch_loader = BatchLoader(params, None)

	top[0].reshape(self.batch_size, 3, params['cur_shape'][0],params['cur_shape'][1])
        top[1].reshape(self.batch_size, 3, params['cur_shape'][0], params['cur_shape'][1])
        top[2].reshape(self.batch_size, 3, params['next_shape'][0], params['next_shape'][1])
        top[3].reshape(self.batch_size, 1, params['next_shape'][0], params['next_shape'][1])

    def forward(self, bottom, top):
	#print 'Doing forward. Queue size: ', self.queue.qsize()
        for itt in range(self.batch_size):
            #im1, im1_masked, im2, label = self.batch_loader.load_next_image()
            item = self.queue.get()

            top[0].data[itt,...] = item['current_image'] #im1
            top[1].data[itt,...] = item['current_mask'] #im1_masked
            top[2].data[itt,...] = item['next_image'] #im2
            top[3].data[itt,...] = item['label'] #label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, params, coco_db):
        self.batch_size = params['batch_size']
	self.coco_db = coco_db
        self.resizeShape1 = params['cur_shape']
        self.resizeShape2 = params['next_shape']
        
        self.video_transformer = Transformer_dist({'transx_param':(0.0,0.0), 'transy_param':(0.0,0.0), 'rot_param':(0.0, 0.0), 
						   'zoomx_param':(1.0, 0.0), 'zoomy_param':(1.0, 0.0), 'shear_param':(0.0, 2)}, {'sigma_range':(.0, .01), 'gamma_range':(.95, 1.05),
						   'contrast_range':(.95, 1.05), 'brightness_sigma':.05, 'mult_rgb_range':(0.95, 1.05),
						   'blur_param':[(.92, 0), (.04, 2), (.03, 3), (.01, 5)]})

	sqrt2 = np.sqrt(2)
        self.frame_transformer = Transformer_dist({'transx_param':(0.0,0.27035499630719317/sqrt2), 'transy_param':(0.0,0.03967734490564475/sqrt2), 'rot_param':(0.0, 5), 
						   'zoomy_param':(1.0, 0.1625625379509229/sqrt2), 'zoomx_param':(1.0, 0.11389299050167503/sqrt2), 'shear_param':(0.0, 1)}, {'sigma_range':(.0, .01), 'gamma_range':(.99, 1.02),
						   'contrast_range':(.98, 1.02), 'brightness_sigma':.01, 'mult_rgb_range':(0.99, 1.01),
						   'blur_param':[(.92, 0), (.04, 2), (.03, 3), (.01, 5)]})
						   
        self.indexes = np.arange(coco_db['length'])
	self.cur = coco_db['length']
	self.shuffle = params['shuffle']
	self.noisy_mask = params['noisy_mask']
	self.bgr = params['bgr']
	self.mean = np.array(params['mean']).reshape(1,1,3)
	
	#Always store mean in RGB format
	self.mean = self.mean[:,:, ::-1]
	
	self.img1_bb_enlargement = params['bb1_enlargment']
	self.img2_bb_enlargement = params['bb2_enlargment']
	self.scale_256 = params['scale_256']
	
	
    def load_next_image(self):
        if self.cur == self.coco_db['length']:
            self.cur = 0
            if self.shuffle:
		random.shuffle(self.indexes)
	
        img_cur, m = read_coco_instance(self.coco_db, self.cur)
        rmin, rmax, cmin, cmax = bbox(m)
        frame1_tran = self.video_transformer.sample()
        frame2_tran = frame1_tran + self.frame_transformer.sample(x_scale = cmax+1-cmin, y_scale = rmax+1-rmin)
        image1 = frame1_tran.transform_img(img_cur.copy(), img_cur.shape[:2], m)
        mask1  = frame1_tran.transform_mask(m.copy(), m.shape)[0]
        mask1[mask1 == -1] = 0        	
        image2 = frame2_tran.transform_img(img_cur.copy(),img_cur.shape[:2], m)
        mask2 = frame2_tran.transform_mask(m.copy(), m.shape)[0]
        mask2[mask2 == -1] = 0
        
        if mask1.sum() < 1000:
	    self.cur += 1
	    return self.load_next_image()
        
        
        # Cropping and resizing
        mask1_bbox = bbox(mask1)
        cimg = crop(image1, mask1_bbox, bbox_enargement_factor = self.img1_bb_enlargement, output_shape = self.resizeShape1, resize_order = 3) - self.mean
        cmask = crop(mask1.astype('float32'), mask1_bbox, bbox_enargement_factor = self.img1_bb_enlargement, output_shape = self.resizeShape1, resize_order = 0)
        if self.noisy_mask:
	    cmask = add_noise_to_mask(cmask)
        cimg_masked = (cimg * cmask[:,:,np.newaxis])
        nimg = crop(image2, mask1_bbox, bbox_enargement_factor = self.img2_bb_enlargement, output_shape = self.resizeShape2, resize_order = 3) - self.mean
        label = crop(mask2.astype('float32'), mask1_bbox, bbox_enargement_factor = self.img2_bb_enlargement, output_shape = self.resizeShape2, resize_order = 0)
        
        self.cur += 1
        
        if self.bgr:
	    cimg = cimg[:,:, ::-1]
	    cimg_masked = cimg_masked[:, :, ::-1]
	    nimg = nimg[:, :, ::-1]
	    
	if self.scale_256:
	    cimg *= 255
	    cimg_masked *= 255
	    nimg *= 255
	    
        item = {'current_image':cimg.transpose((2,0,1)),
                'current_mask' :cimg_masked.transpose((2,0,1)),
                'next_image'   :nimg.transpose((2,0,1)),
                'label'        :label[np.newaxis, :, :]}
	
        return item