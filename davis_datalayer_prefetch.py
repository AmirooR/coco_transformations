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

class DavisDataLayerPrefetch(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['current_image', 'masked_image','next_image','label']
        params = eval(self.param_str)
        check_params(params)
        self.batch_size = params['batch_size']
        self.num_threads = params['num_threads']
        self.max_queue_size = params['max_queue_size']
	self.split = params['split']
        self.queue = Queue(self.max_queue_size)
        self.processes = [None]*self.num_threads
        self.db_info = db_read_info()
        self.sequences = [x for x in self.db_info.sequences if x['set'] == self.split]
        for i in range(self.num_threads):
            batch_loader = BatchLoader(params, self.db_info, self.sequences)
            self.processes[i] = LoaderProcess(kwargs={'queue':self.queue,'loader':batch_loader})
            self.processes[i].daemon = True
            self.processes[i].start()
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

            top[0].data[itt,...] = item['current_image'] #im1
            top[1].data[itt,...] = item['current_mask'] #im1_masked
            top[2].data[itt,...] = item['next_image'] #im2
            top[3].data[itt,...] = item['label'] #label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, params, db_info, sequences):
        self.batch_size = params['batch_size']
        self.resizeShape1 = (params['im_shape'][0]/2, params['im_shape'][1]/2)
        self.resizeShape2 = params['im_shape']
        self.mean = np.array(params['mean'])
        self.db_info = db_info
        self.sequences = sequences

        self.cur1 = 0
        self.cur2 = 0
        self.indexes = np.arange(len(self.sequences))
        random.shuffle(self.indexes)
        self.seqindexes = [np.arange(int(x['num_frames'])-1) for x in self.sequences]
        for x in self.seqindexes:
            random.shuffle(x)

    def load_next_image(self):
        if self.cur2 == len(self.seqindexes[self.indexes[self.cur1]]):
            self.cur1 += 1
            self.cur2 = 0 
        if self.cur1 == len(self.indexes):
            self.cur1 = 0
            for x in self.seqindexes:
                random.shuffle(x)
            random.shuffle(self.indexes)
            self.cur2 = 0 #don't need it
        file_name = osp.join(cfg.PATH.SEQUENCES_DIR, self.sequences[self.indexes[self.cur1]]['name'], '%05d.jpg' % self.seqindexes[self.indexes[self.cur1]][self.cur2])
        uint_image = io.imread(file_name)
        if len(uint_image.shape) == 2:
            tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
            tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
            uint_image = tmp_image
        float_image = np.array(uint_image, dtype=np.float32)/255.0
        #reading mask
        mask_name = osp.join(cfg.PATH.ANNOTATION_DIR, self.sequences[self.indexes[self.cur1]]['name'], '%05d.png' % self.seqindexes[self.indexes[self.cur1]][self.cur2])
        m_uint = io.imread(mask_name)
        m = np.array(m_uint, dtype=np.float32)
        image1 = float_image #
        mask1  = m / 255. #
        if np.sum(mask1) < 200:
            self.cur2 = self.cur2 + 1
            return self.load_next_image()
        image2_name = osp.join(cfg.PATH.SEQUENCES_DIR, self.sequences[self.indexes[self.cur1]]['name'], '%05d.jpg' % (1 + self.seqindexes[self.indexes[self.cur1]][self.cur2]) )
        uint_image2 = io.imread(image2_name)
        if len(uint_image2.shape) == 2:
            tmp_image = np.zeros(uint_image2.shape + (3,), dtype=np.uint8)
            tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image2
            uint_image2 = tmp_image
        image2 =  np.array(uint_image2, dtype=np.float32)/255.0
        rmin, rmax, cmin, cmax = self.bbox(mask1)
        padySize = (rmax+1 - rmin)/2
        padxSize = (cmax+1 - cmin)/2
        image2_padded = np.pad(image2,
                ((padySize,padySize),(padxSize,padxSize),(0,0)),
                mode='constant')
        mask_name2 = osp.join(cfg.PATH.ANNOTATION_DIR, self.sequences[self.indexes[self.cur1]]['name'], '%05d.png' % (1+self.seqindexes[self.indexes[self.cur1]][self.cur2]) )
        m2_uint = io.imread(mask_name2)
        m2 = np.array(m2_uint, dtype=np.float32)
        mask2 = m2 / 255. 
        mask2_padded = np.pad(mask2, ((padySize,padySize),(padxSize,padxSize)),
                mode='constant')
        cropImage1 = image1[rmin:(rmax+1),cmin:(cmax+1),:]
        current_image = resize(cropImage1, self.resizeShape1, mode='nearest') - self.mean
        cropMask1  = mask1[rmin:(rmax+1),cmin:(cmax+1)]
        current_mask = np.zeros_like(current_image)
        resizedMask1 = resize(cropMask1.astype('float32'), self.resizeShape1, mode='nearest')
        current_mask[:,:,0] = current_image[:,:,0] * resizedMask1
        current_mask[:,:,1] = current_image[:,:,1] * resizedMask1
        current_mask[:,:,2] = current_image[:,:,2] * resizedMask1
        
        cropImage2 = image2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize),:] #TODO: check if +1 is needed!
        next_image = resize(cropImage2, self.resizeShape2, mode='nearest') - self.mean
        cropMask2  = mask2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize)]
        label = resize( cropMask2.astype('float32'), self.resizeShape2, mode='nearest')

        #correct the shapes
        current_image = current_image.transpose((2,0,1)) #3xWxH
        current_mask  = current_mask.transpose((2,0,1))
        next_image    = next_image.transpose((2,0,1))
        label         = np.expand_dims(label, axis=0) #1xWxH
        self.cur1 += 1
        item = {'current_image':current_image,
                'current_mask' :current_mask,
                'next_image'   :next_image,
                'label'        : label}
        return item
        

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax


def check_params(params):
    assert 'split' in params.keys(), 'Params must include split: (training|test)'
    required = ['batch_size','im_shape','num_threads','max_queue_size', 'mean']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
