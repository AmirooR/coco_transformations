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
from Queue import Queue
from threading import Thread

#TODO: put these to param_str ?

dataDir='/home/amir/coco' #TODO: set the exact path!
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

saveDir='/home/amir/coco/PythonAPI/data_%s' % dataType #TODO: set the exact path
saveName='%s/instances.pkl' % saveDir

class LoaderThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, verbose=None):
        Thread.__init__(self, group=group, target=target,
                        name=name,verbose=verbose)
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
        check_params(params)
        self.batch_size = params['batch_size']
        self.num_threads = params['num_threads']
        self.max_queue_size = params['max_queue_size']
        self.queue = Queue(self.max_queue_size)
        self.threads = [None]*self.num_threads
        self.dataType=dataType
        self.dataDir=dataDir
        self.annFile=annFile
        self.saveDir=saveDir
        self.saveName=saveName
        self.coco = COCO(self.annFile)
        with open(self.saveName, 'rb') as inp:
            self.anns = pickle.load(inp)
        self.imgs = self.coco.loadImgs([self.anns[i]['image_id'] for i in range(len(self.anns))])
        for i in range(self.num_threads):
            batch_loader = BatchLoader(params, self.coco, self.anns, self.imgs)
            self.threads[i] = LoaderThread(kwargs={'queue':self.queue,'loader':batch_loader})
            self.threads[i].setDaemon(True)
            self.threads[i].start()
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
    def __init__(self, params, coco, anns, imgs):
        self.batch_size = params['batch_size']
        self.dataType=dataType
        self.dataDir=dataDir
        self.annFile=annFile
        self.saveDir=saveDir
        self.saveName=saveName
        self.resizeShape1 = (params['im_shape'][0]/2, params['im_shape'][1]/2)
        self.resizeShape2 = params['im_shape']
        self.video_transformer = Transformer_dist({'translation_range':(0,0), 'rotation_range':(0, 0), 'zoom_range':(1/1.2, 1.2),
                                     'shear_range':(-10, 10)}, {'sigma_range':(.0, .04), 'gamma_range':(.8, 1.2),
                                     'contrast_range':(.8, 1.2), 'brightness_sigma':.15, 'mult_rgb_range':(0.7, 1.4),
                                     'blur_param':[(.8, 0), (.12, 2), (.05, 4), (.03, 8)]})

        self.frame_transformer = Transformer_dist({'translation_range':(-35,35), 'rotation_range':(-20, +20), 'zoom_range':(1/1.2, 1.2),
                                     'shear_range':(-5, 5)}, {'sigma_range':(.0, .02), 'gamma_range':(.9, 1.1),
                                     'contrast_range':(.8, 1.2), 'brightness_sigma':.07, 'mult_rgb_range':(.9, 1.1),
                                     'blur_param':[(.8, 0), (.12, 2), (.05, 4), (.03, 8)]})
        self.coco = coco
        self.anns = anns
        self.imgs = imgs
        self.cur = 0
        #TODO print
        self.indexes = np.arange(len(self.anns))
        random.shuffle(self.indexes)


    def load_next_image(self):
        if self.cur == len(self.anns):
            self.cur = 0
            random.shuffle(self.indexes)
        img_cur = self.imgs[self.indexes[self.cur]]#self.coco.loadImgs([self.anns[self.cur]['image_id']])[0]
        uint_image = io.imread('%s/images/%s/%s' % (self.dataDir,
            self.dataType,img_cur['file_name']))
        if len(uint_image.shape) == 2:
            tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
            tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
            uint_image = tmp_image
        float_image = np.array(uint_image, dtype=np.float32)/255.0
        rle = mask.frPyObjects(self.anns[self.indexes[self.cur]]['segmentation'],
        img_cur['height'], img_cur['width'])
        m_uint = mask.decode(rle)
        m = np.array(m_uint[:,:,0], dtype=np.float32)
        base_tran = self.video_transformer.sample()
        frame1_tran = base_tran
        frame2_tran = base_tran + self.frame_transformer.sample()
        image1 = frame1_tran.transform_img(float_image.copy(),
                float_image.shape[:2], m)
        mask1  = frame1_tran.transform_mask(m.copy(), m.shape)

        mask1 = mask1[0]
        mask1[mask1 == -1] = 0
        if np.sum(mask1) < 200:
            self.cur = self.cur + 1
            return self.load_next_image()
        image2 = frame2_tran.transform_img( float_image.copy(),
                float_image.shape[:2], m)
        rmin, rmax, cmin, cmax = self.bbox(mask1)
        padySize = (rmax+1 - rmin)/2
        padxSize = (cmax+1 - cmin)/2
        image2_padded = np.pad(image2,
                ((padySize,padySize),(padxSize,padxSize),(0,0)),
                mode='constant')
        mask2 = frame2_tran.transform_mask(m.copy(), m.shape)
        mask2 = mask2[0]
        mask2[mask2 == -1] = 0
        mask2_padded = np.pad(mask2, ((padySize,padySize),(padxSize,padxSize)),
                mode='constant')
        cropImage1 = image1[rmin:(rmax+1),cmin:(cmax+1),:]
        current_image = resize(cropImage1, self.resizeShape1, mode='nearest')
        cropMask1  = mask1[rmin:(rmax+1),cmin:(cmax+1)]
        current_mask = np.zeros_like(current_image)
        resizedMask1 = resize(cropMask1.astype('float32'), self.resizeShape1, mode='nearest')
        current_mask[:,:,0] = current_image[:,:,0] * resizedMask1
        current_mask[:,:,1] = current_image[:,:,1] * resizedMask1
        current_mask[:,:,2] = current_image[:,:,2] * resizedMask1
        
        cropImage2 = image2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize),:] #TODO: check if +1 is needed!
        next_image = resize(cropImage2, self.resizeShape2, mode='nearest')
        cropMask2  = mask2_padded[rmin:(rmin+4*padySize),
                cmin:(cmin+4*padxSize)]
        label = resize( cropMask2.astype('float32'), self.resizeShape2, mode='nearest')

        #correct the shapes
        current_image = current_image.transpose((2,0,1)) #3xWxH
        current_mask  = current_mask.transpose((2,0,1))
        next_image    = next_image.transpose((2,0,1))
        label         = np.expand_dims(label, axis=0) #1xWxH
        self.cur += 1
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
    required = ['batch_size','im_shape','num_threads','max_queue_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
