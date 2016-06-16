import caffe

from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=0, dilation=1, param_name=None):
    conv_param = {'kernel_size':ks}
    if stride != 1:
        conv_param['stride'] = stride
    if pad != 0:
        conv_param['pad'] = pad
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

def simple_net(split, initialize_fc8=False):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892), seed=1337)

    if split == 'train':
        pydata_params['sbdd_dir'] = '../../data/sbdd/dataset'
        pylayer = 'SBDDSegDataLayer'
        n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module='voc_layers', layer=pylayer, ntop=4, param_str=str(pydata_params))
    elif split == 'val':
        pydata_params['voc_dir'] = '../../data/pascal/VOC2011'
        pylayer = 'VOCSegDataLayer'
        n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module='voc_layers', layer=pylayer, ntop=4, param_str=str(pydata_params))
    elif split == 'deploy':
         n.cur_im, n.label_1 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
         n.masked_im, n.label_2 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
         n.next_im, n.label_3 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    else:
        raise Exception
   
            
    

    # current image vgg-net
    n.conv1_1c, n.relu1_1c = conv_relu(n.cur_im, 64, pad=100, param_name='conv1_1')
    n.conv1_2c, n.relu1_2c = conv_relu(n.relu1_1c, 64, pad=1, param_name='conv1_2')
    n.pool1c = max_pool(n.relu1_2c)

    n.conv2_1c, n.relu2_1c = conv_relu(n.pool1c, 128, pad=1, param_name='conv2_1')
    n.conv2_2c, n.relu2_2c = conv_relu(n.relu2_1c, 128, pad=1, param_name='conv2_2')
    n.pool2c = max_pool(n.relu2_2c)

    n.conv3_1c, n.relu3_1c = conv_relu(n.pool2c, 256, pad=1, param_name='conv3_1')
    n.conv3_2c, n.relu3_2c = conv_relu(n.relu3_1c, 256, pad=1, param_name='conv3_2')
    n.conv3_3c, n.relu3_3c = conv_relu(n.relu3_2c, 256, pad=1, param_name='conv3_3')
    n.pool3c = max_pool(n.relu3_3c)

    n.conv4_1c, n.relu4_1c = conv_relu(n.pool3c, 512, pad=1, param_name='conv4_1')
    n.conv4_2c, n.relu4_2c = conv_relu(n.relu4_1c, 512, pad=1, param_name='conv4_2')
    n.conv4_3c, n.relu4_3c = conv_relu(n.relu4_2c, 512, pad=1, param_name='conv4_3')

    n.conv5_1c, n.relu5_1c = conv_relu(n.relu4_3c, 512, pad=2, dilation=2, param_name='conv5_1')
    n.conv5_2c, n.relu5_2c = conv_relu(n.relu5_1c, 512, pad=2, dilation=2, param_name='conv5_2')
    n.conv5_3c, n.relu5_3c = conv_relu(n.relu5_2c, 512, pad=2, dilation=2, param_name='conv5_3')


    # masked image vgg-net
    n.conv1_1m, n.relu1_1m = conv_relu(n.masked_im, 64, pad=100, param_name='conv1_1')
    n.conv1_2m, n.relu1_2m = conv_relu(n.relu1_1m, 64, pad=1, param_name='conv1_2')
    n.pool1m = max_pool(n.relu1_2m)

    n.conv2_1m, n.relu2_1m = conv_relu(n.pool1m, 128, pad=1, param_name='conv2_1')
    n.conv2_2m, n.relu2_2m = conv_relu(n.relu2_1m, 128, pad=1, param_name='conv2_2')
    n.pool2m = max_pool(n.relu2_2m)

    n.conv3_1m, n.relu3_1m = conv_relu(n.pool2m, 256, pad=1, param_name='conv3_1')
    n.conv3_2m, n.relu3_2m = conv_relu(n.relu3_1m, 256, pad=1, param_name='conv3_2')
    n.conv3_3m, n.relu3_3m = conv_relu(n.relu3_2m, 256, pad=1, param_name='conv3_3')
    n.pool3m = max_pool(n.relu3_3m)

    n.conv4_1m, n.relu4_1m = conv_relu(n.pool3m, 512, pad=1, param_name='conv4_1')
    n.conv4_2m, n.relu4_2m = conv_relu(n.relu4_1m, 512, pad=1, param_name='conv4_2')
    n.conv4_3m, n.relu4_3m = conv_relu(n.relu4_2m, 512, pad=1, param_name='conv4_3')

    n.conv5_1m, n.relu5_1m = conv_relu(n.relu4_3m, 512, pad=2, dilation=2, param_name='conv5_1')
    n.conv5_2m, n.relu5_2m = conv_relu(n.relu5_1m, 512, pad=2, dilation=2, param_name='conv5_2')
    n.conv5_3m, n.relu5_3m = conv_relu(n.relu5_2m, 512, pad=2, dilation=2, param_name='conv5_3')

    # next image vgg-net
    n.conv1_1n, n.relu1_1n = conv_relu(n.next_im, 64, pad=100, param_name='conv1_1')
    n.conv1_2n, n.relu1_2n = conv_relu(n.relu1_1n, 64, pad=1, param_name='conv1_2')
    n.pool1n = max_pool(n.relu1_2n)

    n.conv2_1n, n.relu2_1n = conv_relu(n.pool1n, 128, pad=1, param_name='conv2_1')
    n.conv2_2n, n.relu2_2n = conv_relu(n.relu2_1n, 128, pad=1, param_name='conv2_2')
    n.pool2n = max_pool(n.relu2_2n)

    n.conv3_1n, n.relu3_1n = conv_relu(n.pool2n, 256, pad=1, param_name='conv3_1')
    n.conv3_2n, n.relu3_2n = conv_relu(n.relu3_1n, 256, pad=1, param_name='conv3_2')
    n.conv3_3n, n.relu3_3n = conv_relu(n.relu3_2n, 256, pad=1, param_name='conv3_3')
    n.pool3n = max_pool(n.relu3_3n)

    n.conv4_1n, n.relu4_1n = conv_relu(n.pool3n, 512, pad=1, param_name='conv4_1')
    n.conv4_2n, n.relu4_2n = conv_relu(n.relu4_1n, 512, pad=1, param_name='conv4_2')
    n.conv4_3n, n.relu4_3n = conv_relu(n.relu4_2n, 512, pad=1, param_name='conv4_3')
    
    n.conv5_1n, n.relu5_1n = conv_relu(n.relu4_3n, 512, pad=2, dilation=2, param_name='conv5_1')
    n.conv5_2n, n.relu5_2n = conv_relu(n.relu5_1n, 512, pad=2, dilation=2, param_name='conv5_2')
    n.conv5_3n, n.relu5_3n = conv_relu(n.relu5_2n, 512, pad=2, dilation=2, param_name='conv5_3')
    
    # concatination
    n.concat1 = L.Concat(n.relu5_3c, n.relu5_3m, n.relu5_3n)
    #n.concat1 = n.relu5_3n
    
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
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='xavier'), num_output=2)
        else:
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=2)
        
    
    n.upscore = L.Deconvolution(n.fc8, convolution_param=dict(kernel_size=16, stride=8, num_output=2, weight_filler=dict(type='bilinear'),
                                                              bias_term=False), param=dict(lr_mult=0, decay_mult=0))

    n.score = crop(n.upscore, n.next_im)
    
    if split != 'deploy':
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(normalize=False, ignore_label=255))
    else:
        n.prop = L.Softmax(n.score)
    return n

def save_net(net_path, proto):
    with open(net_path, 'w') as f:
        f.write(proto)

if __name__ == '__main__':
    phases = ['train', 'val', 'deploy']
    for phase in phases:
        net_spec = simple_net(phase)
        save_net(phase + '.prototxt', str(net_spec.to_proto()))
    #net = caffe.Net(net_path, caffe.TRAIN)