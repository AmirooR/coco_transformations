import cv2
import numpy as np
import sys
import skimage.io as io

def read_flo_file(file_path):
    """
    reads a flo file, it is for little endian architectures,
    first slice, i.e. data2D[:,:,0], is horizontal displacements
    second slice, i.e. data2D[:,:,1], is vertical displacements

    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            cprint('Magic number incorrect. Invalid .flo file: %s' % file_path, bcolors.FAIL)
            raise  Exception('Magic incorrect: %s !' % file_path)
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            data2D = np.reshape(data, (h, w, 2), order='C')
            return data2D

def reflow( img1, img2, flow):
    mymap = np.indices(flow.shape[:2])
    map_t = mymap.transpose(1,2,0)
    new_map = flow + map_t[:,:,::-1]
    newFrame = cv2.remap(img2, new_map[:,:,0].astype('float32'), new_map[:,:,1].astype('float32'), cv2.INTER_LINEAR)
    return newFrame, new_map

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print 'Usage: %s img1 img2 flo' % sys.argv[0]
        sys.exit()
    
    img1 = io.imread(sys.argv[1])
    img2 = io.imread(sys.argv[2])
    flow = read_flo_file(sys.argv[3])
    
    newFrame, newmap = reflow(img1, img2, flow)
    io.imsave('newFrame.png', newFrame)
    io.imsave('newFrame-img1.png', np.abs(newFrame - img1) )
    print 'img1 sum: ', np.abs(newFrame - img1)
    
