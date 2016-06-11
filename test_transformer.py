from matplotlib.pylab import imshow, figure, title
import pylab as pl
from skimage import data
from transformer import Transformer_dist
import numpy as np
video_transformer = Transformer_dist({'translation_range':(-20,+20), 'rotation_range':(-20, 20), 'zoom_range':(1/1.2, 1.2), 
                                      'shear_range':(-10, 10)}, {'sigma_range':(.0, .04), 'gamma_range':(.8, 1.2), 
                                      'contrast_range':(.8, 1.2), 'brightness_sigma':.15, 'mult_rgb_range':(0.7, 1.4), 
                                      'blur_param':[(1, 4)]})
                                              
frame_transformer = Transformer_dist({'translation_range':(-10,10), 'rotation_range':(-10, +10), 'zoom_range':(1/1.1, 1.1), 
                                      'shear_range':(-5, 5)}, {'sigma_range':(.0, .02), 'gamma_range':(.9, 1.1),
                                      'contrast_range':(.8, 1.2), 'brightness_sigma':.07, 'mult_rgb_range':(.80, 1.20), 
                                      'blur_param':[(.8, 0), (.15, 2), (.05, 4)]})
                                      


uint_image = data.astronaut() 
float_image = np.array(uint_image, dtype=np.float32) / 255.0
#mask = np.array(float_image[:,:,0] > .5, dtype=np.float32)
mask = np.zeros_like(float_image[:,:,0], dtype=np.float32)
mask[100:350,100:350] = 1.
imshow(float_image)
title('Original Image')

base_tran = video_transformer.sample()
frame1_tran = base_tran + frame_transformer.sample()
frame2_tran = base_tran + frame_transformer.sample()

image1 = frame1_tran.transform_img(float_image.copy(), float_image.shape[:2],
        mask)
mask1 = frame1_tran.transform_mask(mask.copy(), mask.shape)
#fills padded area with -1
mask1 = mask1[0]
mask1[mask1 == -1] = 0

image2 = frame2_tran.transform_img(float_image.copy(), float_image.shape[:2])
mask2 = frame2_tran.transform_mask(mask.copy(), mask.shape)
mask2 = mask2[0]
mask2[mask2 == -1] = 0

figure()
imshow(image1)
title('Image 1')
figure()
imshow(mask1)
title('Mask 1')
figure()
imshow(image2)
title('Image 2')
figure()
imshow(mask2)
title('Mask 2')
pl.show()
