import numpy as np
import scipy.io as spio
import os
import abc
import skimage.io as skio

class ToDavisConverter(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

    def get_images_dir(self):
        return self.images_dir

    def get_annotations_dir(self):
        return self.annotations_dir

    def get_sequence_names(self):
        seq_list = []
        l = os.listdir(self.images_dir)
        for x in l:
            file_path  = os.path.join(self.images_dir, x)
            if os.path.isdir(file_path):
                seq_list.append(x)
        return seq_list

    @abc.abstractmethod
    def get_annotated_frames_list(self, seq_name):
        pass

    @abc.abstractmethod
    def get_number_of_objects(self, seq_name): 
        pass
    
    def get_id_of_objects(self, seq_name): 
        pass
    
    
    @abc.abstractmethod
    def read_segmentation(self, seq_name, frame_file_name, annotator_idx = 0):
        pass

    @abc.abstractmethod
    def convert_seq_to_davis(self, wrt_images_dir, wrt_annotations_dir,
            seq_name, wrt_dir_prefix, set_name, yml_file):
        pass 

    def convert_all_seqs_to_davis(self, wrt_images_dir, wrt_annotations_dir,
            wrt_dir_prefix, set_name, yml_file): #NOTE: yml_file should be opened beforehand
        seq_list = self.get_sequence_names()
        for x in seq_list:
            self.convert_seq_to_davis(wrt_images_dir, wrt_annotations_dir, x,
                    wrt_dir_prefix, set_name, yml_file)

class VSBToDavisConverter(ToDavisConverter):
    def __init__(self,images_dir, annotations_dir, read_obj_file = False):
        super(VSBToDavisConverter, self).__init__(images_dir, annotations_dir)
        self.read_obj_file = read_obj_file
    
    def get_annotated_frames_list(self, seq_name):
        seq_dir = os.path.join(self.annotations_dir, seq_name)
        tmp_list = os.listdir(seq_dir)
        frames_list = []
        for x in sorted(tmp_list):
            if x.endswith('.mat'):
                frames_list.append(x)
        return frames_list

    def get_id_of_objects(self, seq_name): 
	obj_file_path = os.path.join(self.annotations_dir, seq_name, 'obj.txt')
	print 'reading object list from', obj_file_path
	return np.loadtxt(obj_file_path,delimiter=',').reshape(-1, 4)
    
    def get_number_of_objects(self, seq_name): 
        nobj = 0
        frames_list = self.get_annotated_frames_list(seq_name)
        for frame in frames_list:
            segmentation = self.read_segmentation(seq_name, frame)
            nobj = max(nobj, np.max(segmentation) )
        return nobj

    def read_segmentation(self, seq_name, frame_file_name, annotator_idx=0):
        annotation_file = os.path.join(self.annotations_dir, seq_name,
                frame_file_name)
        annotation = spio.loadmat( annotation_file)
        segmentation = annotation['groundTruth'][0,annotator_idx]['Segmentation'][0,0]
        return segmentation
    
    def convert_obj_to_davis(self, frames_list, seq_name, images_dir_path, annotations_dir_path, obj_idx): 
	frame_idx = 0
	for frame_file_name in frames_list:
	    segmentation = self.read_segmentation( seq_name, frame_file_name)
            mask = np.zeros(segmentation.shape, dtype='uint8')
            mask[ np.where( segmentation == obj_idx) ] = 255 #TODO: check
            if mask.sum() == 0:
		continue
	    frame_idx += 1
	    image_file_name = frame_file_name[:-3] + 'jpg'
            image_file = os.path.join(self.images_dir, seq_name,
                        image_file_name)
            image = skio.imread(image_file)
            wrt_image_name = '%05d.jpg' % frame_idx
            wrt_image_path = os.path.join( images_dir_path, wrt_image_name)
            skio.imsave( wrt_image_path, image)
            wrt_annotation_name = '%05d.png' % frame_idx
            wrt_annotation_path = os.path.join( annotations_dir_path,
                        wrt_annotation_name)
            skio.imsave( wrt_annotation_path, mask)
              
    def convert_seq_to_davis(self, wrt_images_dir, wrt_annotations_dir,
            seq_name, wrt_dir_prefix, set_name, yml_file):
        
        frames_list = self.get_annotated_frames_list(seq_name)
        
        if self.read_obj_file:
	    obj_idxs = self.get_id_of_objects(seq_name)[:, 0]
	else:
	    nobj = self.get_number_of_objects(seq_name)
	    obj_idxs = np.arange(nobj) + 1
        for obj_idx in obj_idxs:
            dir_name = wrt_dir_prefix + '-' + seq_name + '-%03d' % obj_idx
            images_dir_path = os.path.join(wrt_images_dir, dir_name)
            annotations_dir_path = os.path.join(wrt_annotations_dir, dir_name)
            if not os.path.exists(images_dir_path):
                os.makedirs(images_dir_path)
            if not os.path.exists(annotations_dir_path):
                os.makedirs(annotations_dir_path)
            yml_file.write('- attributes: []\n')
            yml_file.write('  name: %s\n' % dir_name)
            yml_file.write('  num_frames: %d\n' % len(frames_list) )
            yml_file.write('  set: %s\n' % set_name)
            self.convert_obj_to_davis(frames_list, seq_name, images_dir_path, annotations_dir_path, obj_idx)


class SegTrackToDavisConverter(ToDavisConverter):
    def __init__(self,images_dir, annotations_dir):
        super(SegTrackToDavisConverter, self).__init__(images_dir, annotations_dir)
    
    def get_annotated_frames_list(self, seq_name):
        seq_dir = os.path.join(self.images_dir, seq_name)
        tmp_list = os.listdir(seq_dir)
        frames_list = []
        for x in sorted(tmp_list):
            if x.endswith('.png') or x.endswith('.bmp'):
                frames_list.append(x)
        return frames_list

    def get_number_of_objects(self, seq_name):#NOTE: don't write anything in dataset folders
        images_path = os.path.join(self.annotations_dir, seq_name)
        l = os.listdir(images_path)
        first_file = os.path.join(images_path, l[0])
        if os.path.isdir(first_file):
            return len(l)
        else:
            return 1
        
    def read_segmentation(self, seq_name, frame_file_name, annotator_idx=0):
        pass
        
    def convert_seq_to_davis(self, wrt_images_dir, wrt_annotations_dir,
            seq_name, wrt_dir_prefix, set_name, yml_file):
        nobj = self.get_number_of_objects(seq_name)
        frames_list = self.get_annotated_frames_list(seq_name)
        for i in range(nobj):
            obj_idx = i + 1
            dir_name = wrt_dir_prefix + '-' + seq_name + '-%03d' % obj_idx
            images_dir_path = os.path.join(wrt_images_dir, dir_name)
            annotations_dir_path = os.path.join(wrt_annotations_dir, dir_name)
            if not os.path.exists(images_dir_path):
                os.makedirs(images_dir_path)
            if not os.path.exists(annotations_dir_path):
                os.makedirs(annotations_dir_path)
            yml_file.write('- attributes: []\n')
            yml_file.write('  name: %s\n' % dir_name)
            yml_file.write('  num_frames: %d\n' % len(frames_list) )
            yml_file.write('  set: %s\n' % set_name)
            for frame_idx, frame_file_name in enumerate(frames_list):
                if nobj == 1:
                    ann_file = os.path.join(self.annotations_dir, seq_name,
                            frame_file_name)
                else:
                    ann_file = os.path.join(self.annotations_dir, seq_name,
                            str(obj_idx), frame_file_name)
                if not os.path.isfile(ann_file): #NOTE it's a tokhmi way to check
                    ann_file = ann_file[:-3] + 'png'
                    if not os.path.isfile(ann_file):
                        ann_file = ann_file[:-3] + 'bmp'
                        if not os.path.isfile(ann_file):
                            ann_file = ann_file[:-3] + 'jpg'
                mask = skio.imread(ann_file)
                image_file_name = frame_file_name
                image_file = os.path.join(self.images_dir, seq_name,
                        image_file_name)
                image = skio.imread(image_file)
                wrt_image_name = '%05d.jpg' % frame_idx
                wrt_image_path = os.path.join( images_dir_path, wrt_image_name)
                skio.imsave( wrt_image_path, image)
                wrt_annotation_name = '%05d.png' % frame_idx
                wrt_annotation_path = os.path.join( annotations_dir_path,
                        wrt_annotation_name)
                skio.imsave( wrt_annotation_path, mask)

