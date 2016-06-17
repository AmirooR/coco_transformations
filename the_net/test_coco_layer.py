import caffe
import time
import numpy as np
from matplotlib.pyplot import imshow, show, figure
from skimage import io
new_path = 'test_coco_layer.prototxt'    
new_net = caffe.Net(new_path, caffe.TRAIN)


for i in range(10):
  print 'iteration', i 
  if i == 2:
      time.sleep(20)
  new_net.forward()
  cur_im = new_net.blobs['cur_im'].data
  masked_im = new_net.blobs['masked_im'].data
  next_im = new_net.blobs['next_im'].data
  label = new_net.blobs['label'].data
  name = '%d_0cur_im.png' % i
  io.imsave(name, (cur_im[0].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_1cur_im.png' % i
  io.imsave(name, (cur_im[1].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_0masked_im.png' % i
  io.imsave(name, (masked_im[0].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_1masked_im.png' % i
  io.imsave(name, (masked_im[1].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_0next_im.png' % i
  io.imsave(name, (next_im[0].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_1next_im.png' % i
  io.imsave(name, (next_im[1].transpose((1,2,0))*255).astype('uint8'))
  name = '%d_0label.png' % i 
  io.imsave(name, (label[0][0]*255).astype('uint8'))
  name = '%d_1label.png' % i
  io.imsave(name, (label[1][0]*255).astype('uint8'))
  
