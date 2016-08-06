import util
import os.path as osp
from davis import cfg

def get_epicflow_input_lists(split):
    sequences = util.load_davis_sequences(split)
    img1_list = []
    img2_list = []
    flow_list = []
    for sequence in sequences:
        orig_name = sequence['name']
        for frame_id in xrange(sequence['num_frames'] - 1):
            img1_list.append( cfg.PATH.SEQUENCES_DIR, orig_name, '%05d.jpg' % frame_id)
            img2_list.append( cfg.PATH.SEQUENCES_DIR, orig_name, '%05d.jpg' % frame_id+1)
            flow_list.append( cfg.PATH.ANNOTATION_DIR, orig_name, '%05d_inv.flo' % frame_id+1)

    return img1_list, img2_list, flow_list
