import caffe
import numpy as np
from net import conv_relu, max_pool, save_net, simple_net
from os import remove
from caffe import layers as L, params

def transplant(new_net, net, suffix=''):
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def create_dilated_net():
    n = caffe.NetSpec()
    n.im, n.label_1 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)

    n.conv1_1, n.relu1_1 = conv_relu(n.im, 64, pad=100, param_name='conv1_1')
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, pad=1, param_name='conv1_2')
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, pad=1, param_name='conv2_1')
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, pad=1, param_name='conv2_2')
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, pad=1, param_name='conv3_1')
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, pad=1, param_name='conv3_2')
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, pad=1, param_name='conv3_3')
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, pad=1, param_name='conv4_1')
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, pad=1, param_name='conv4_2')
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, pad=1, param_name='conv4_3')

    n.conv5_1, n.relu5_1 = conv_relu(n.relu4_3, 512, pad=2, dilation=2, param_name='conv5_1')
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, pad=2, dilation=2, param_name='conv5_2')
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, pad=2, dilation=2, param_name='conv5_3')
    
    # fully conv
    n.fc6, n.relu6 = conv_relu(n.conv5_3, 4096, ks=7, dilation=4)        
    n.fc7, n.relu7 = conv_relu(n.relu6, 4096, ks=1, pad=0)
    n.fc8 = L.Convolution(n.relu7, kernel_size=1, num_output=21)
    
    return n


if __name__ == '__main__':
    dilated_path = 'dilated_net_tmp.prototxt'
    dilated_weights = 'dilation8_pascal_voc.caffemodel'
    new_weights = 'simple_net.caffemodel'
    new_path = 'new_net_tmp.prototxt'
    
    dilated_netspec = create_dilated_net()
    new_netspec = simple_net('deploy', initialize_fc8=True)  
    save_net(dilated_path, str(dilated_netspec.to_proto()))
    save_net(new_path, str(new_netspec.to_proto()))
    dilated_net = caffe.Net(dilated_path, dilated_weights, caffe.TRAIN)
    new_net = caffe.Net(new_path, caffe.TRAIN)
    #transplant vgg-net conv weights
    transplant(new_net, dilated_net, 'c')
    #transplant fc6 weights
    new_net.params['fc6'][0].data[:, -512:][...] = dilated_net.params['fc6'][0].data
    new_net.params['fc6'][1].data[...] = dilated_net.params['fc6'][1].data
    #transplant fc7 weights
    new_net.params['fc7'][0].data[...] = dilated_net.params['fc7'][0].data
    new_net.params['fc7'][1].data[...] = dilated_net.params['fc7'][1].data
    new_net.save(new_weights)
    remove(dilated_path)
    remove(new_path)