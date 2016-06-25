import json
import time
import caffe
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from transformer import Transformer_dist
from skimage.transform import resize
from multiprocessing import Process, Queue, Pool
from davis import cfg
from davis.dataset.utils import db_read_info
import os.path as osp
import zmq
import sys
import multiprocessing

def unwarp_batch_loader_load_frame(args,**kwargs):
    return BatchLoader.load_frame(*args,**kwargs)

class LoaderProcess(Process):
    def __init__(self, name=None, args=(),
                 kwargs=None):
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
	self.db_info = kwargs['db_info']
        self.firstTime = True
 
	
    def run(self):
        #Initiate socket
	self.context = zmq.Context.instance()
	self.sock = self.context.socket(zmq.REP)
	self.sock.connect('tcp://localhost:' + self.port)
        print bcolors.OKBLUE, 'Server started', bcolors.ENDC

	mask_predicted = np.zeros((self.batch_size, ) + self.prediction_shape)
	mask_crop_param = [None] * self.batch_size
        while True:
	    #check if we have a message to read
	    if not self.firstTime: #np.any(self.cur_frame_indices != 0):
	    	mask_predicted = self.read_message()
            	    
	    for i in xrange(self.batch_size):
		if mask_predicted[i].sum() == 0 and self.cur_frame_indices[i] != 0:
                    self.cur_frame_indices[i] = sys.maxint
		    print bcolors.WARNING + 'OBJECT DISAPEARED! SEQUENCE WILL BE CHANGED' + bcolors.ENDC
                    
		self.update_indices(i)
		print '>'*10, 'cur_frame_indices =', self.cur_frame_indices[i]
            pool = multiprocessing.pool.ThreadPool(self.batch_size)
            seqs = [self.sequences[self.cur_seq_indices[i]]['name'] for i in range(self.batch_size)]
            
		#item = self.loader.load_frame(self.sequences[self.cur_seq_indices[i]]['name'], self.cur_frame_indices[i], mask_predicted[i], mask_crop_param[i])
            items = pool.map(unwarp_batch_loader_load_frame, zip([self.loader]*self.batch_size, seqs, self.cur_frame_indices, mask_predicted, mask_crop_param))
            pool.close()
            pool.join()
            for i in range(self.batch_size):
		mask_crop_param[i] = items[i]['label_crop_param']
		self.queue.put(items[i])
            self.firstTime = False

    def update_indices(self, i):
        if self.cur_seq_indices[i] < 0 or self.cur_frame_indices[i] >= (self.sequences[i]['num_frames']-2):
            self.cur_seq_indices[i] = self.next_seq_index
            self.next_seq_index = (self.next_seq_index + 1) % len(self.sequences)
            self.cur_frame_indices[i] = 0
        else:
            self.cur_frame_indices[i] += 1
		
    def read_message(self):
        print bcolors.WARNING + 'server before receive' + bcolors.ENDC
	message = self.sock.recv_pyobj()
        print bcolors.WARNING + 'server received message ' + str(message.shape) + bcolors.ENDC
        self.sock.send('OK')
        print 'ok sent'
       	return message
		
class DavisDataLayerServer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['current_image', 'masked_image','next_image','label']
        params = eval(self.param_str)
        check_params(params)
        self.batch_size = params['batch_size']
	self.split = params['split']
        self.port = params['port']
        self.queue = Queue(self.batch_size)
        self.db_info = db_read_info()
        self.sequences = [x for x in self.db_info.sequences if x['set'] == self.split]
        batch_loader = BatchLoader(params)
        self.process = LoaderProcess(kwargs={'queue':self.queue,'loader':batch_loader,'batch_size':self.batch_size,'prediction_shape':params['im_shape'],'port':self.port,'db_info':self.db_info,'sequences':self.sequences})
        self.process.daemon = True
        self.process.start()
        #self.batch_loader = BatchLoader(params, None)

        top[0].reshape(
                self.batch_size, 3, params['im_shape'][0]/2,
                params['im_shape'][1]/2) #NOTE: current is devided by 2
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
	#print 'Doing forward. Queue size: ', self.queue.qsize()
        for itt in range(self.batch_size):
            #im1, im1_masked, im2, label = self.batch_loader.load_next_image()
            item = self.queue.get()
            print bcolors.OKGREEN +  str(item['label'].sum()) + ' ya khoda' + bcolors.ENDC

            top[0].data[itt,...] = item['current_image'] #im1
            top[1].data[itt,...] = item['current_masked'] #im1_masked
            top[2].data[itt,...] = item['next_image'] #im2
            top[3].data[itt,...] = item['label'] #label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.resizeShape1 = (params['im_shape'][0]/2, params['im_shape'][1]/2)
        self.resizeShape2 = params['im_shape']
    
    
	
    def load_frame(self, seq_name, frame, mask1_cropped, mask1_crop_param):
        
        # reading first image
        file_name = osp.join(cfg.PATH.SEQUENCES_DIR, seq_name, '%05d.jpg' % frame)
        uint_image = io.imread(file_name)
        if len(uint_image.shape) == 2:
            tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
            tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
            uint_image = tmp_image
        float_image = np.array(uint_image, dtype=np.float32)/255.0
        print bcolors.WARNING + 'FRAME = ' + str(frame) + bcolors.ENDC
        image1 = float_image #

        #reading mask
        if frame == 0:
            mask_name = osp.join(cfg.PATH.ANNOTATION_DIR, seq_name, '%05d.png' % frame)
            m_uint = io.imread(mask_name)
            m = np.array(m_uint, dtype=np.float32)
            mask1  = m / 255. #
        else:
	    #convert mask1 to its original shape using mask1_crop_param
	    mask1 = self.get_original_mask(mask1_cropped, mask1_crop_param, float_image.shape[0:2])
	
	#reading second image
        image2_name = osp.join(cfg.PATH.SEQUENCES_DIR, seq_name, '%05d.jpg' % (1 + frame) )
	uint_image2 = io.imread(image2_name)
        if len(uint_image2.shape) == 2:
            tmp_image = np.zeros(uint_image2.shape + (3,), dtype=np.uint8)
            tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image2
            uint_image2 = tmp_image
        image2 =  np.array(uint_image2, dtype=np.float32)/255.0
        
        #reading second mask
        mask_name2 = osp.join(cfg.PATH.ANNOTATION_DIR, seq_name, '%05d.png' % (1+frame) )
        m2_uint = io.imread(mask_name2)
        m2 = np.array(m2_uint, dtype=np.float32)
        mask2 = m2 / 255. 
        
        
        #cropping
        rmin, rmax, cmin, cmax = self.bbox(mask1)
        padySize = (rmax+1 - rmin)/2
        padxSize = (cmax+1 - cmin)/2
        image2_padded = np.pad(image2,
                ((padySize,padySize),(padxSize,padxSize),(0,0)),
                mode='constant')
        
        mask2_padded = np.pad(mask2, ((padySize,padySize),(padxSize,padxSize)),
                mode='constant')
        cropImage1 = image1[rmin:(rmax+1),cmin:(cmax+1),:]
        current_image = resize(cropImage1, self.resizeShape1, mode='nearest')
        cropMask1  = mask1[rmin:(rmax+1),cmin:(cmax+1)]
        current_masked = np.zeros_like(current_image)
        resizedMask1 = resize(cropMask1.astype('float32'), self.resizeShape1, mode='nearest')
        current_masked[:,:,0] = current_image[:,:,0] * resizedMask1
        current_masked[:,:,1] = current_image[:,:,1] * resizedMask1
        current_masked[:,:,2] = current_image[:,:,2] * resizedMask1
        
        cropImage2 = image2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize),:]
        next_image = resize(cropImage2, self.resizeShape2, mode='nearest')
        cropMask2  = mask2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize)]
        label = resize( cropMask2.astype('float32'), self.resizeShape2, mode='nearest')

        #correct the shapes
        current_image = current_image.transpose((2,0,1)) #3xWxH
        current_masked  = current_masked.transpose((2,0,1))
        next_image    = next_image.transpose((2,0,1))
        label         = np.expand_dims(label, axis=0) #1xWxH
        label_crop_param = np.array([padySize, padxSize, rmin, (rmin+4*padySize), cmin, (cmin+4*padxSize)])

        item = {'current_image': current_image,
                'current_masked' : current_masked,
                'current_mask' : resizedMask1,
                'next_image'   :next_image,
                'label'        : label,
                'label_crop_param' : label_crop_param}
        return item

            
    def get_original_mask(self, mask_cropped, crop_param, orig_size):
        new_size = (orig_size[0] + 2*crop_param[0], orig_size[1] + 2*crop_param[1])
	orig_mask = np.zeros(new_size, dtype=np.float32)
	mask_resized = resize(mask_cropped, [crop_param[3] - crop_param[2], crop_param[5] - crop_param[4]], mode='nearest')
	orig_mask[crop_param[2]:(crop_param[3]), crop_param[4]:(crop_param[5])] = mask_resized	
	return orig_mask[(crop_param[0] ):(orig_size[0] + crop_param[0]), (crop_param[1] ):(orig_size[1] + crop_param[1] )]
    
    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax


def check_params(params):
    assert 'split' in params.keys(), 'Params must include split: (training|test)'
    required = ['batch_size','im_shape','port']
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
