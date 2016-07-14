import numpy as np
import skimage
import skimage.io
import skimage.transform
from skimage.morphology import disk
from skimage.filters import rank
import random


#class TransformerParam:
#    def __init__(self, transform_param=None, color_adjustment_param=None):
#        self.transform_params = [transform_param]
#        self.color_adjustment_params = [color_adjustment_param]
    
    #    def __iadd__(self, other):
    #    self.transform_params += other.number
#    return self


class Transformer_dist:
    def __init__(self, transform_param=None, color_adjustment_param=None):
        self.transform_param = transform_param
        self.color_adjustment_param = color_adjustment_param
        
    def sample(self, x_scale = 1.0, y_scale = 1.0):
        det_transform_param = self.random_perturbation(x_scale, y_scale, **self.transform_param)
        det_color_adj_param = self.random_color_perturbation(**self.color_adjustment_param)
        return Transformer(det_transform_param, det_color_adj_param)

    def random_color_perturbation(self, sigma_range = (.0, .0), gamma_range = (1.0, 1.0), contrast_range = (1.0, 1.0), brightness_sigma = .0, mult_rgb_range = (1.0, 1.0), blur_param = [(1,0)]):
        sigma = np.random.uniform(*sigma_range)
        gamma = np.random.uniform(*gamma_range)
        contrast = np.random.uniform(*contrast_range)
        brightness = np.random.randn() * brightness_sigma
        mult_rgb = np.random.uniform(*mult_rgb_range, size=[1, 3])
        
        blur_probs, blur_vals = zip(*blur_param)
        bins = np.add.accumulate(blur_probs)
        if bins[-1] != 1:
            raise Exception
        blur_radius =  blur_vals[np.digitize(np.random.random_sample(), bins)]

        return {'gaussian_std':sigma, 'gamma':gamma, 'contrast':contrast, 'brightness':brightness, 'mult_rgb':mult_rgb, 'blur_radius': blur_radius}
        
    def random_perturbation(self, x_scale, y_scale, zoomx_param=(1.0, 0.0), zoomy_param = (1.0, 0.0),  rot_param=(0.0, 0.0), shear_param=(0.0, 0.0), transx_param=(0.0,0.0), transy_param=(0.0, 0.0), do_flip=False):
        translation_x = 0.0 if transx_param[1] <= 0 else np.random.laplace(*transx_param) * x_scale
        translation_y = 0.0 if transy_param[1] <= 0 else np.random.laplace(*transy_param) * y_scale
        rotation = 0.0 if rot_param[1] <= 0 else np.random.laplace(*rot_param)
        shear = 0.0 if shear_param[1] <= 0 else np.random.laplace(*shear_param)
        
        if do_flip and (np.random.randint(2) > 0):
            shear += 180
            rotation += 180
      
        zoom_x = 1.0 if zoomx_param[1] <= 0 else np.random.laplace(*zoomx_param)
        zoom_y = 1.0 if zoomx_param[1] <= 0 else np.random.laplace(*zoomy_param)
        
        return {'zoom':np.array([zoom_x, zoom_y]), 'rotation':rotation, 'shear':shear, 'translation':np.array([translation_x, translation_y])}
    
    def __str__(self):
        ss = str(self.transform_param) + ' ' + str(self.color_adjustment_param)
        return ss
# Supports rotation, zoom, translation, gaussian noise, contrast changes, multiplicative color changes, gamma value changes
class Transformer:
    def __init__(self, transform_param=None, color_adjustment_param=None):
        self.transform_param = transform_param
        self.color_adjustment_param = color_adjustment_param
            
    def __add__(self, other):         
        if other == None:
            return Transformer(self.transform_param, self.color_adjustment_param)
        
        transform_param = {}
        transform_param['zoom'] = self.transform_param['zoom'] * other.transform_param['zoom']
        transform_param['rotation'] = self.transform_param['rotation'] + other.transform_param['rotation']
        transform_param['shear'] = self.transform_param['shear'] + other.transform_param['shear']
        transform_param['translation'] = self.transform_param['translation'] + other.transform_param['translation']
        
        
        color_adjustment_param = {}
        color_adjustment_param['gaussian_std'] = (self.color_adjustment_param['gaussian_std'] ** 2 + other.color_adjustment_param['gaussian_std'] ** 2) ** .5
        color_adjustment_param['gamma'] = self.color_adjustment_param['gamma'] * other.color_adjustment_param['gamma']
        color_adjustment_param['contrast'] = self.color_adjustment_param['contrast'] * other.color_adjustment_param['contrast']
        color_adjustment_param['brightness'] = self.color_adjustment_param['brightness'] + other.color_adjustment_param['brightness']
        color_adjustment_param['mult_rgb'] = self.color_adjustment_param['mult_rgb'] * other.color_adjustment_param['mult_rgb']
        color_adjustment_param['blur_radius'] = max(self.color_adjustment_param['blur_radius'], other.color_adjustment_param['blur_radius'])
        return Transformer(transform_param, color_adjustment_param)
            
    def __radd__(self, other):
        if other == None:
            return Transformer(self.transform_param, self.color_adjustment_param)        
        return other.__add__(self)
        
    def transform(self, img, mask, final_size):
        #print img.max(), img.min(), mask.max(), mask.min()
        img = self.transform_img(img, final_size)
        mask = self.transform_mask(mask, final_size)
        #print img.max(), img.min(), mask.max(), mask.min()
        #skimage.io.imsave('mask' + str(np.random.random_integers(0, 100)) + '.png', np.array(mask, 'float64')/mask.max())
        return (img, mask)

    def transform_img(self, img, final_size, mask=None):
        if self.transform_param == None:
            trans_perturbation = skimage.transform.AffineTransform()
        else:
            trans_perturbation = self.build_augmentation_transform(image_height=img.shape[0], image_width=img.shape[1], **self.transform_param)
        
        if self.color_adjustment_param != None:
            img = self.color_adjustment(img=img, mask=mask, **self.color_adjustment_param)
        
        img = self.fast_warp(img, trans_perturbation, cval=0)
        img = self.correct_ratio(img, final_size, cval=0)
        img = skimage.transform.resize(img, final_size, order=1, mode='constant', cval=0)
        return img
    
    def transform_mask(self, orig_mask, final_size):
        if self.transform_param == None:
            trans_perturbation = skimage.transform.AffineTransform()
        else:
            trans_perturbation = self.build_augmentation_transform(image_height=orig_mask.shape[0], image_width=orig_mask.shape[1], **self.transform_param)
        
        mask = self.fast_warp(orig_mask, trans_perturbation, order=0, cval=-1)
        mask = self.correct_ratio(mask, final_size, cval=-1)
        scale = final_size[0]/float(mask.shape[0])
        mask = np.array(skimage.transform.resize(mask, final_size, order=0, mode='constant', cval=-1), dtype='int')
        
        
        #compute scale and occlusion variables:
        points = np.array([[0,0],[mask.shape[1], 0],[0, orig_mask.shape[0]],[orig_mask.shape[1], orig_mask.shape[0]]], dtype=np.int)
        tpoints = trans_perturbation.inverse(points)
        min_vals = np.round(np.min(tpoints, 0)) - [10, 10]
        tform_points = skimage.transform.SimilarityTransform(translation=min_vals)
        new_trans = tform_points + trans_perturbation
        bbox_size = (np.max(tpoints, 0) - min_vals) + [10, 10]
        complete_mask = self.fast_warp(orig_mask, new_trans, output_shape = [bbox_size[1], bbox_size[0]], order=0, cval=-1)        
        new_size = tuple(np.array(np.array(complete_mask.shape) * scale, dtype=np.int))
        complete_mask = np.array(skimage.transform.resize(complete_mask, new_size, order=0, mode='constant', cval=-1), dtype='int') 
        return (mask, complete_mask)
        
    def color_adjustment(self, img, mask=None, gaussian_std=.0, gamma=1.0, contrast = 1.0, brightness = 0, mult_rgb = np.array([1.0, 1.0, 1.0]), blur_radius = 0):
        img **= gamma
        img *= contrast
        img += np.random.randn(*img.shape).astype('float32') * gaussian_std
        img += brightness
        img *= mult_rgb
        np.clip(img, 0.0, 1.0, img)
        blur_mask = None
        if mask is not None:
            blur_mask = random.choice([mask,1-mask,np.ones_like(mask)])
        if blur_radius > 0:
            selem = disk(blur_radius)
            tmp_img = img.copy()
            for i in range(img.shape[2]):
                img[:, :, i] = rank.mean(img[:, :, i], selem=selem,mask=blur_mask) / 255.0
            img[np.where(blur_mask == 0)] = tmp_img[np.where(blur_mask==0)]

        return img
    
    def fast_warp(self, img, tf, output_shape=None, order = 1, mode='constant', cval=0):
        """
            This wrapper function is about five times faster than skimage.transform.warp, for our use case.
            """
        m = tf.params
        
        if output_shape is None:
            output_shape = tuple(img.shape[0:2])
            
        if len(img.shape) == 2:
            img_wf = np.empty(output_shape, dtype='float64')
            img_wf[...] = skimage.transform._warps_cy._warp_fast(img, m, order=order, mode=mode, cval=cval, output_shape=output_shape)
        else:   
            channels = img.shape[2]
            img_wf = np.empty(output_shape + (channels,), dtype='float64')
            for k in xrange(channels):
                img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, order=order, mode=mode, cval=cval, output_shape=output_shape)
        
        return img_wf

    def correct_ratio(self, array, ratio, cval):
#        if ratio[0] < array.shape[0] or ratio[1] < array.shape[1]:
#            r1 = array.shape[0] / float(ratio[0])         
#            r2 = array.shape[1] / float(ratio[1])
#            if r1 > r2:
#                big_size = (array.shape[0], int(ratio[1] * r1))
#            else:
#                big_size = (int(ratio[0] * r2), array.shape[1])
#        else:
#            big_size = ratio
        
        r1 = array.shape[0] / float(ratio[0])         
        r2 = array.shape[1] / float(ratio[1])
        if r1 > r2:
            big_size = (array.shape[0], int(ratio[1] * r1))
        else:
            big_size = (int(ratio[0] * r2), array.shape[1])
                
        start = (int((big_size[0] - array.shape[0])/2), int((big_size[1] - array.shape[1])/2))
        end = (start[0] + array.shape[0], start[1] + array.shape[1])
        
            
        #padd image and the mask with respect to the new image size
        if len(array.shape) == 3:
            new_array = np.ones(big_size + (array.shape[2],)) * cval
        else:
            new_array = np.ones(big_size) * cval
        
        new_array[start[0]:end[0], start[1]:end[1]] = array
        return new_array
    
    
    
    
    def build_augmentation_transform(self, image_height, image_width, zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0)):
	zoom = np.array(zoom)
        center_shift = np.array((image_height, image_height)) / 2. - 0.5
        tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
    
        tform_augment = skimage.transform.AffineTransform(scale=1.0/zoom, rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
        tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
        return tform
