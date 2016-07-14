import numpy as np
from skimage.transform import resize
from davis.dataset.utils import db_read_info
from davis import cfg
import os.path as osp
import skimage.io as io
from scipy import misc
import random
import math
from pycocotools.coco import COCO
from pycocotools import mask
from skimage.morphology import disk
from skimage.filters import rank
import pickle

debug_mode = False
def cprint(string, style = None):
    if not debug_mode:
	return
    if style is None:
	print str(string)
    else:
	print style + str(string) + bcolors.ENDC

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def add_noise_to_mask(cmask, r_param = (20, 20), mult_param = (20, 5), threshold = .2):
    radius = max(np.random.normal(*r_param), 1)
    mult = max(np.random.normal(*mult_param), 2)
    
    selem = disk(radius)
    mask2d = np.zeros(cmask.shape + (2,))
    mask2d[:, :, 0] = rank.mean((1 - cmask).copy(), selem=selem) / 255.0
    mask2d[:, :, 1] = rank.mean(cmask.copy(), selem=selem) / 255.0

    exp_fmask = np.exp(mult * mask2d);
    max_fmask = exp_fmask[:,:,1] / np.sum(exp_fmask, 2);
    max_fmask[max_fmask < threshold] = 0;
    
    return max_fmask
    
def bbox(img):
    if img.sum() == 0:
	raise Exception('Input mask can not be empty!')
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop(img, bbox, bbox_enargement_factor = 1, constant_pad = 0, output_shape = None, resize_order = 1):
    pad_size = ((bbox_enargement_factor - 1) * np.array([bbox[1]+1 - bbox[0], bbox[3]+1 - bbox[2]], 'float32')/2.0 + constant_pad).astype('int')
    pad_param = tuple(zip(pad_size, pad_size)) + (len(img.shape) - 2) * ((0,0),)
    img_padded = np.pad(img, pad_param, mode='constant')
    output = img_padded[bbox[0]:(bbox[1] + 2 * pad_size[0] + 1), bbox[2]:(bbox[3] + 2 * pad_size[1] + 1)]
    if output_shape is not None:
	return resize( output, output_shape, order = resize_order, mode = 'nearest')
    else:
	return output

def crop_undo(img, bbox, bbox_enargement_factor = 1, constant_pad = 0, output_shape = None, resize_order = 1):
    if len(img.shape) != 2:
	raise Exception
    pad_size = ((bbox_enargement_factor - 1) * np.array([bbox[1]+1 - bbox[0], bbox[3]+1 - bbox[2]], 'float32')/2.0 + constant_pad).astype('int')
    correct_size = np.array([bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1]) + 2 * pad_size
    img = resize(img, correct_size, order = resize_order, mode = 'nearest')
    if output_shape is not None:
	output = np.zeros((output_shape[0] + 2 * pad_size[0], output_shape[1] + 2 * pad_size[1]))
	output[bbox[0]:(bbox[1] + 2 * pad_size[0] + 1), bbox[2]:(bbox[3] + 2 * pad_size[1] + 1)] = img
	return output[pad_size[0]:-(1+pad_size[0]), pad_size[1]:-(1+pad_size[1])]
    else:
	return img
    
#defaults is a list of (key, val) is val is None key is required field
def check_params(params, **kwargs):
    for key, val in kwargs.items():
	key_defined = (key in params.keys())
	if val is None:
	    assert key_defined, 'Params must include {}'.format(key)
	elif not key_defined:
	    params[key] = val

def load_netflow_db(annotations_file, split, shuffle = False):
    if split == 'training':
        split = 1
    if split == 'test':
        split = 2
    annotations = np.loadtxt(annotations_file)
    frame_indices = np.arange(len(annotations))
    frame_indices = frame_indices[ annotations == split ]
    data_dir = osp.join(osp.dirname(osp.abspath(annotations_file)), 'data/')
    if shuffle:
        random.shuffle( frame_indices)

    return dict(frame_indices=frame_indices, data_dir=data_dir)

def read_netflow_instance(netflow_db, instance_id):
    data_dir = netflow_db['data_dir']
    instance_id = instance_id + 1
    img1 = io.imread( osp.join(data_dir, '%05d_img1.ppm' % instance_id))
    img2 = io.imread( osp.join(data_dir, '%05d_img2.ppm' % instance_id))
    flow = read_flo_file( osp.join(data_dir, '%05d_flow.flo' % instance_id))
    return img1, img2, flow

def read_coco_instance(coco_db, instance_id, load_mask = True):
    ann = coco_db['coco'].loadAnns(coco_db['anns'][instance_id])[0]
    img_cur = coco_db['coco'].loadImgs(ann['image_id'])[0]
    uint_image = io.imread('%s/images/%s/%s' % (coco_db['dataDir'],
                            coco_db['dataType'],img_cur['file_name']))
    if len(uint_image.shape) == 2:
    	tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
    	tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
        uint_image = tmp_image
    float_image = np.array(uint_image, dtype=np.float32)/255.0
    
    if load_mask:
	rle = mask.frPyObjects(ann['segmentation'], img_cur['height'], img_cur['width'])
	m_uint = mask.decode(rle)
	m = np.array(m_uint[:, :, 0], dtype=np.float32)
	return (float_image, m)
    return float_image
    
def load_coco_db(dataDir, dataType, cats=[], areaRng=[], iscrowd=False, shuffle = False, chunck_num = 0):
    if dataType == 'training':
	dataType = 'train2014'
    elif dataType == 'test':
	dataType = 'val2014'
    else:
	raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')
    annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
    
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=cats);
    anns = coco.getAnnIds(catIds=catIds, areaRng=areaRng, iscrowd=iscrowd)
    cprint(str(len(anns)) + ' annotations read from coco', bcolors.OKGREEN)
    if shuffle:
	random.shuffle(anns)
    if chunck_num == 0:
	return dict(coco=coco, dataDir=dataDir, dataType=dataType, annFile=annFile, cats=cats, areaRng=areaRng, length=len(anns), anns=anns)
    else:
	ind = range(0, len(anns), int(len(anns) / chunck_num) + 1)
	ind.append(len(anns))
	return [dict(coco=coco, dataDir=dataDir, dataType=dataType, annFile=annFile, cats=cats, areaRng=areaRng, length=ind[i+1] - ind[i], anns=anns[ind[i]:ind[i+1]]) for i in xrange(len(ind) - 1)]
    
    
    
   
#def read_pascal_instance(pascal_db, instance_id, load_mask = True):
    #ann = pascal_db['anns'][instance_id]
    #img_path = 
    #uint_image = io.imread('%s/images/%s/%s' % (coco_db['dataDir'],
                            #coco_db['dataType'],img_cur['file_name']))
    #if len(uint_image.shape) == 2:
    	#tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
    	#tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
        #uint_image = tmp_image
    #float_image = np.array(uint_image, dtype=np.float32)/255.0
    
    #if load_mask:
	#mobj_path = osp.join(self.db_path, 'SegmentationObject', pascal_db[ + '.png')
	#mobj_uint = misc.imread(mobj_path)
	#m_uint = mask.decode(rle)
	#m = np.array(m_uint[:, :, 0], dtype=np.float32)
	#return (float_image, m)
    #return float_image
    
#def load_pascal_db(dataDir, dataType, cats=[], areaRng=[], shuffle = False, chunck_num = 0):
    #if dataType == 'training':
	#dataType = 'train'
    #elif dataType == 'test':
	#dataType = 'val'
    #else:
	#raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')
    
    #pascal = PASCAL(dataDir, dataType)
    #catIds = pascal.getCatIds(catNms=cats)
    #anns = p.getAnns(catIds=catIds, areaRng=areaRng)
    #getAnns(self, catIds=[], areaRng=[0, np.inf]):
    #cprint(str(len(anns)) + ' annotations read from pascal', bcolors.OKGREEN)
    #if shuffle:
	#random.shuffle(anns)
    #if chunck_num == 0:
	#return dict(pascal=pascal, dataDir=dataDir, dataType=dataType, cats=cats, areaRng=areaRng, length=len(anns), anns=anns)
    #else:
	#ind = range(0, len(anns), int(len(anns) / chunck_num) + 1)
	#ind.append(len(anns))
	#return [dict(pascal=pascal, dataDir=dataDir, dataType=dataType, cats=cats, areaRng=areaRng, length=ind[i+1] - ind[i], anns=anns[ind[i]:ind[i+1]]) for i in xrange(len(ind) - 1)]



def read_davis_frame(sequences, seq_id, frame_id, load_mask = True):
    assert seq_id < len(sequences)
    assert frame_id < sequences[seq_id]['num_frames']
    orig_name = sequences[seq_id]['name']
    start_id = sequences[seq_id]['start_id']
    file_name = osp.join(cfg.PATH.SEQUENCES_DIR, orig_name, '%05d.jpg' % (frame_id + start_id))
    uint_image = io.imread(file_name)
    if len(uint_image.shape) == 2:
	tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
        tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
        uint_image = tmp_image
    image = np.array(uint_image, dtype=np.float32)/255.0
    if load_mask:
	mask_name = osp.join(cfg.PATH.ANNOTATION_DIR, orig_name, '%05d.png' % (frame_id + start_id))
	m_uint = io.imread(mask_name)
	mask = np.array(m_uint, dtype=np.float32) / 255.0
	return (image, mask)
    return image
    
# 1376 Test
# 2079 Training
def load_davis_sequences(split, max_seq_len = 0, shuffle = False):
    assert split == 'training' or split == 'test', 'split could be in this format: (training|test)'
    db_info = db_read_info()
    full_sequences = [x for x in db_info.sequences if x['set'] == split]
    for sequence in full_sequences:
	sequence['start_id'] = 0
    
    
    if max_seq_len > 0:
	assert max_seq_len > 1
	splited_sequences = list()
	
	for sequence in full_sequences:
	    frame_num = sequence['num_frames']
	    start_ids = range(0, frame_num - 1, max_seq_len)
	    frame_nums = [max_seq_len] * len(start_ids)
	    frame_nums[-1] =  frame_num - start_ids[-1]
	    #cprint(frame_num, bcolors.WARNING)
	    #cprint(str(frame_nums), bcolors.WARNING)
	    #cprint(str(start_ids), bcolors.WARNING)
	    splited_sequence = [sequence] * len(start_ids)
	    map(dict.__setitem__, splited_sequence, ['start_id'] * len(start_ids), start_ids)
	    map(dict.__setitem__, splited_sequence, ['num_frames'] * len(frame_nums), frame_nums)
	    splited_sequences.extend(splited_sequence)
	cprint(str(splited_sequences), bcolors.WARNING)
	full_sequences = splited_sequences
	if shuffle:
	    random.shuffle(full_sequences)
	
    return full_sequences
    
    
class PASCAL:
    def __init__(self, db_path, dataType):
	self.db_path = db_path
	classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
	self.name_id_map = dict(zip(classes, range(1, len(classes) + 1)))
	self.dataType = dataType
    def getCatIds(self, catNms=[]):
	return [self.name_id_map[catNm] for catNm in catNms]
    
    def get_annotation_path(self):
	return osp.join(self.db_path, self.dataType + '_anns.pkl')
    
    def load_annotations(self):
	path = self.get_annotation_path()
	if not osp.exists(path):
	    self.create_annotations()
	
	with open(path, 'rb') as f:
	    anns = pickle.load(f)
	return anns
    
    def create_annotations(self):
	with open(osp.join(self.db_path, 'ImageSets', 'Segmentation', self.dataType + '.txt'), 'r') as f:
	    names = f.readlines()    
	    names = [name[:-1] for name in names]
	anns = []
	for item in names:
	    mobj_path = osp.join(self.db_path, 'SegmentationObject', item + '.png')
	    mobj_uint = misc.imread(mobj_path)
	    obj_ids, obj_sizes = np.unique(mobj_uint, return_counts=True)
	    mclass_path = osp.join(self.db_path, 'SegmentationClass', item + '.png')
	    mclass_uint = misc.imread(mclass_path)
	    
	    for obj_idx in xrange(len(obj_ids)):
		class_id = int(np.median(mclass_uint[mobj_uint == obj_ids[obj_idx]]))
		if class_id == 0 or class_id == 255 or obj_ids[obj_idx] == 0 or obj_ids[obj_idx] == 255:
		    continue
		anns.append(dict(image_name=item, mask_name=item, object_id=obj_idx+1, class_id=class_id, object_size = obj_sizes[obj_idx]))
	
	with open(self.get_annotation_path(), 'w') as f:
	    pickle.dump(anns, f)
    
    #def readAnn(ann):
	#image_path = 
	#mask_path = 
	#obect_id = 
	#return (image_path, mask_path, object_id
    #Each annotion has image_name, mask_name, object_id, class_id, object_size keys
    def getAnns(self, catIds=[], areaRng=[]):
	
	if areaRng == []:
	    areaRng = [0, np.inf]
	anns = self.load_annotations()
	if catIds == []:
	    if areaRng == [0, np.inf]:
		return anns
	    catIds = self.getCatIds(self.nam_id_map.keys())
	
	filtered_anns = [ann for ann in anns if ann['class_id'] in catIds and areaRng[0] < ann['object_size'] and ann['object_size'] < areaRng[1]]
	return filtered_anns
