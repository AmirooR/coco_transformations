import caffe

from caffe import layers as L, params as P
from caffe.coord_map import coord_map_from_to
import numpy as np

def conv_relu(bottom, nout, ks=3, stride=1, pad=0, dilation=1, param_name=None):
    conv_param = {'kernel_size':ks}
    if stride != 1:
        conv_param['stride'] = stride
    if not hasattr(pad, '__len__'):
	if pad != 0:
	    conv_param['pad'] = int(pad)
    else:
	if pad[0] != 0:
	    conv_param['pad_w'] = int(pad[0])
	if pad[1] != 0:
	    conv_param['pad_h'] = int(pad[1])
    if dilation != 1:
        conv_param['dilation'] = dilation
    
    conv_param['num_output'] = nout
    if param_name is None:
         conv_param['param']= [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        conv_param['param']= [dict(name=param_name + '_w', lr_mult=1, decay_mult=1), dict(name=param_name + '_b', lr_mult=2, decay_mult=0)]
    conv = L.Convolution(bottom, **conv_param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def conv_vgg(n, im, suffix='', last_layer_pad=0, first_layer_pad=0):
  
    conv, relu = conv_relu(im, 64, pad=first_layer_pad, param_name='conv1_1')
    setattr(n, 'conv1_1' + suffix, conv)
    setattr(n, 'relu1_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 64, pad=1, param_name='conv1_2')
    setattr(n, 'conv1_2' + suffix, conv)
    setattr(n, 'relu1_2' + suffix, relu)
    
    pool = max_pool(relu)
    setattr(n, 'pool1' + suffix, pool)
    
    
    
    conv, relu = conv_relu(pool, 128, pad=1, param_name='conv2_1')
    setattr(n, 'conv2_1' + suffix, conv)
    setattr(n, 'relu2_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 128, pad=1, param_name='conv2_2')
    setattr(n, 'conv2_2' + suffix, conv)
    setattr(n, 'relu2_2' + suffix, relu)
    
    pool = max_pool(relu)
    setattr(n, 'pool2' + suffix, pool)
    
    
    conv, relu = conv_relu(pool, 256, pad=1, param_name='conv3_1')
    setattr(n, 'conv3_1' + suffix, conv)
    setattr(n, 'relu3_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 256, pad=1, param_name='conv3_2')
    setattr(n, 'conv3_2' + suffix, conv)
    setattr(n, 'relu3_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 256, pad=1, param_name='conv3_3')
    setattr(n, 'conv3_3' + suffix, conv)
    setattr(n, 'relu3_3' + suffix, relu)
    pool = max_pool(relu)
    setattr(n, 'pool3' + suffix, pool)
    

    conv, relu = conv_relu(pool, 512, pad=1, param_name='conv4_1')
    setattr(n, 'conv4_1' + suffix, conv)
    setattr(n, 'relu4_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=1, param_name='conv4_2')
    setattr(n, 'conv4_2' + suffix, conv)
    setattr(n, 'relu4_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=1, param_name='conv4_3')
    setattr(n, 'conv4_3' + suffix, conv)
    setattr(n, 'relu4_3' + suffix, relu)


    conv, relu = conv_relu(relu, 512, pad=2, dilation=2, param_name='conv5_1')
    setattr(n, 'conv5_1' + suffix, conv)
    setattr(n, 'relu5_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=2, dilation=2, param_name='conv5_2')
    setattr(n, 'conv5_2' + suffix, conv)
    setattr(n, 'relu5_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=2+last_layer_pad, dilation=2, param_name='conv5_3')
    setattr(n, 'conv5_3' + suffix, conv)
    setattr(n, 'relu5_3' + suffix, relu)
    
    return n
  
def simple_net(split, initialize_fc8=False, cur_shape = None, next_shape = None, batch_size=1,  num_threads=1, max_queue_size=5):
  
    #Get crop layer parameters
    tmp_net = caffe.NetSpec()
    tmp_net.im, tmp_net.label = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    conv_vgg(tmp_net, tmp_net.im, suffix='', last_layer_pad=0, first_layer_pad=100)
    tmp_net.fc6, tmp_net.relu6 = conv_relu(tmp_net.conv5_3, 4096, ks=7, dilation=4)        
    tmp_net.fc7, tmp_net.relu7 = conv_relu(tmp_net.relu6, 4096, ks=1, pad=0)
    tmp_net.fc8 = L.Convolution(tmp_net.relu7, kernel_size=1, num_output=2)
    tmp_net.upscore = L.Deconvolution(tmp_net.fc8, convolution_param=dict(kernel_size=16, stride=8, num_output=2))
    
    ax, a, b = coord_map_from_to(tmp_net.upscore, tmp_net.im)
    assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
    assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
    assert (np.round(b) == b).all(), 'cannot crop noninteger offset (b = {})'.format(b)
    #
    
    #Create network
    n = caffe.NetSpec()

    if split == 'train':
	pydata_params = dict(batch_size=batch_size, im_shape=tuple(next_shape), num_threads=num_threads, max_queue_size=max_queue_size)
        n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module='coco_transformed_datalayers_prefetch', layer='CocoTransformedDataLayerPrefetch', ntop=4, param_str=str(pydata_params))
    elif split == 'val':
	pydata_params = dict(batch_size=batch_size, im_shape=tuple(next_shape), num_threads=num_threads, max_queue_size=max_queue_size)
        n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module='coco_transformed_datalayers_prefetch', layer='CocoTransformedDataLayerPrefetch', ntop=4, param_str=str(pydata_params))
    elif split == 'deploy':
         n.cur_im, n.label_1 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
         n.masked_im, n.label_2 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
         n.next_im, n.label_3 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    else:
        raise Exception
   
    
    if cur_shape is None or next_shape is None:
	concat_pad = np.zeros((2,))
    else:
      concat_pad = (next_shape - cur_shape)/2.0/8.0
    if not all(concat_pad == np.round(concat_pad)):
	raise Exception

    

    conv_vgg(n, n.cur_im, suffix='c', last_layer_pad=concat_pad, first_layer_pad=100)
    conv_vgg(n, n.masked_im, suffix='m', last_layer_pad=concat_pad, first_layer_pad=100)
    conv_vgg(n, n.next_im, suffix='n', last_layer_pad=0, first_layer_pad=100)
    
    # concatination
    n.concat1 = L.Concat(n.relu5_3c, n.relu5_3m, n.relu5_3n)
    
    # fully conv
    n.fc6, n.relu6 = conv_relu(n.concat1, 4096, ks=7, dilation=4)        
    if split == 'train':
        n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
        n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
        n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
        n.fc8 = L.Convolution(n.drop7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=2)
    else:
        n.fc7, n.relu7 = conv_relu(n.relu6, 4096, ks=1, pad=0)
        if initialize_fc8:
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian', std=.01), num_output=2)
        else:
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=2)
        
    
    n.upscore = L.Deconvolution(n.fc8, convolution_param=dict(kernel_size=16, stride=8, num_output=2, group=2, weight_filler=dict(type='bilinear'),
                                                              bias_term=False), param=dict(lr_mult=0, decay_mult=0))

    n.score = L.Crop(n.upscore, n.next_im,
                  crop_param=dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int))))
    
    if split != 'deploy':
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(ignore_label=255))
    else:
        n.prop = L.Softmax(n.score)
    return n

def save_net(net_path, proto):
    with open(net_path, 'w') as f:
        f.write(proto)

if __name__ == '__main__':
    phases = ['train', 'val', 'deploy']
    next_shape = np.array([384,384])
    batch_size = 3
    num_threads = 5
    max_queue_size = 20
    for phase in phases:
        net_spec = simple_net(phase, cur_shape=next_shape/2, next_shape=next_shape, batch_size=3, num_threads=num_threads, max_queue_size=max_queue_size)
        save_net(phase + '.prototxt', str(net_spec.to_proto()))
    #net = caffe.Net(net_path, caffe.TRAIN)
