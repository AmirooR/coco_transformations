from pycocotools.coco import COCO
import numpy as np
import cPickle as pickle
import random
import skimage.io as io
from pycocotools import mask
from transformer import Transformer_dist

dataDir='..'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

saveDir='data_%s' % dataType
saveName='%s/instances.pkl' % saveDir

imgSaveDir='save_%s' % dataType

padSize = 30

# making transformers
#video_transformer = Transformer_dist({'translation_range':(-20,+20), 'rotation_range':(-20, 20), 'zoom_range':(1/1.2, 1.2), 
#                                      'shear_range':(-10, 10)}, {'sigma_range':(.0, .04), 'gamma_range':(.8, 1.2), 
#                                      'contrast_range':(.8, 1.2), 'brightness_sigma':.15, 'mult_rgb_range':(0.7, 1.4), 
#                                      'blur_param':[(1, 0)]})
#                                              
#frame_transformer = Transformer_dist({'translation_range':(-10,10), 'rotation_range':(-10, +10), 'zoom_range':(1/1.1, 1.1), 
#                                      'shear_range':(-5, 5)}, {'sigma_range':(.0, .02), 'gamma_range':(.9, 1.1),
#                                      'contrast_range':(.8, 1.2), 'brightness_sigma':.07, 'mult_rgb_range':(.80, 1.20), 
#                                      'blur_param':[(.8, 0), (.15, 2), (.05, 4)]})
video_transformer = Transformer_dist({'translation_range':(0,0), 'rotation_range':(0, 0), 'zoom_range':(1/1.2, 1.2),
                                     'shear_range':(-10, 10)}, {'sigma_range':(.0, .04), 'gamma_range':(.8, 1.2),
                                     'contrast_range':(.8, 1.2), 'brightness_sigma':.15, 'mult_rgb_range':(0.7, 1.4),
                                     'blur_param':[(.8, 0), (.12, 2), (.05, 4), (.03, 8)]})

frame_transformer = Transformer_dist({'translation_range':(-35,35), 'rotation_range':(-20, +20), 'zoom_range':(1/1.2, 1.2),
                                     'shear_range':(-5, 5)}, {'sigma_range':(.0, .02), 'gamma_range':(.9, 1.1),
                                     'contrast_range':(.8, 1.2), 'brightness_sigma':.07, 'mult_rgb_range':(.9, 1.1),
                                     'blur_param':[(.8, 0), (.12, 2), (.05, 4), (.03, 8)]})
# initialize coco
coco=COCO(annFile)

# loading instance annotations
with open(saveName,'rb') as inp:
	anns = pickle.load(inp)

# loading image ids corresponding to anns
imgs = coco.loadImgs([anns[i]['image_id'] for i in range(len(anns))])

#loading the counter
try:
    _count = int(open(".counter").read())
except IOError:
    _count = 0

def incrcounter():
    global _count
    _count = _count + 1

def savecounter():
    open(".counter", "w").write("%d" % _count)

import atexit
atexit.register(savecounter)

def bbox_padded(img):
    global padSize
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin-padSize, rmax+padSize+1, cmin-padSize, cmax+padSize+1

for i in np.arange(_count, len(anns)):
    print 'transforming instance %d' % i
    #transform_and_save_image(i)
    uint_image = io.imread('%s/images/%s/%s' % (dataDir,dataType,imgs[i]['file_name']))
    float_image = np.array(uint_image, dtype=np.float32)/255.0
    rle = mask.frPyObjects(anns[i]['segmentation'], imgs[i]['height'], imgs[i]['width'])
    m_uint = mask.decode(rle)
    m = np.array(m_uint[:,:,0], dtype=np.float32)
    base_tran = video_transformer.sample()
    frame1_tran = base_tran # + frame_transformer.sample()
    frame2_tran = base_tran + frame_transformer.sample()
    image1 = frame1_tran.transform_img(float_image.copy(), float_image.shape[:2], m)
    image1_padded = np.pad(image1,((padSize,padSize),(padSize,padSize),(0,0)), mode='constant')
    mask1 = frame1_tran.transform_mask(m.copy(), m.shape)
	
    #fills padded area with -1
    mask1 = mask1[0]
    mask1[mask1 == -1] = 0
    mask1_padded = np.pad(mask1,((padSize,padSize),(padSize,padSize)), mode='constant')

    image2 = frame2_tran.transform_img(float_image.copy(), float_image.shape[:2], m)
    image2_padded = np.pad(image2,((padSize,padSize),(padSize,padSize),(0,0)), mode='constant')
    mask2 = frame2_tran.transform_mask(m.copy(), m.shape)
    mask2 = mask2[0]
    mask2[mask2 == -1] = 0
    mask2_padded = np.pad(mask2,((padSize,padSize),(padSize,padSize)), mode='constant')

    #find crop values
    (rmin, rmax, cmin, cmax) = bbox_padded(mask1)

    #crop
    cropImage1 = image1_padded[rmin:rmax,cmin:cmax,:]
    cropImage2 = image2_padded[rmin:rmax,cmin:cmax,:]
    cropMask1 = mask1_padded[rmin:rmax,cmin:cmax]
    cropMask2 = mask2_padded[rmin:rmax,cmin:cmax]

    #save
    img1saveName = '%s/image1/%d.jpg' % (imgSaveDir, i)
    img2saveName = '%s/image2/%d.jpg' % (imgSaveDir, i)
    mask1saveName = '%s/mask1/%d.png' % (imgSaveDir, i)
    mask2saveName = '%s/mask2/%d.png' % (imgSaveDir, i)
    tran1saveName = '%s/transform1/%d.pkl' % (imgSaveDir, i)
    tran2saveName = '%s/transform2/%d.pkl' % (imgSaveDir, i)
    cropImg1saveName = '%s/cropimage1/%d.jpg' % (imgSaveDir, i)
    cropImg2saveName = '%s/cropimage2/%d.jpg' % (imgSaveDir, i)
    cropMask1saveName = '%s/cropmask1/%d.png' % (imgSaveDir, i)
    cropMask2saveName = '%s/cropmask2/%d.png' % (imgSaveDir, i)

    io.imsave(img1saveName, (image1*255).astype('uint8'))
    io.imsave(img2saveName, (image2*255).astype('uint8'))
    io.imsave(mask1saveName, (mask1*255).astype('uint8'))
    io.imsave(mask2saveName, (mask2*255).astype('uint8'))
    io.imsave(cropImg1saveName, (cropImage1*255).astype('uint8'))
    io.imsave(cropImg2saveName, (cropImage2*255).astype('uint8'))
    io.imsave(cropMask1saveName, (cropMask1*255).astype('uint8'))
    io.imsave(cropMask2saveName, (cropMask2*255).astype('uint8'))
        
    with open(tran1saveName, 'wb') as output:
        pickle.dump(frame1_tran, output, -1)
    with open(tran2saveName, 'wb') as output:
        pickle.dump(frame2_tran, output, -1)
    incrcounter()
