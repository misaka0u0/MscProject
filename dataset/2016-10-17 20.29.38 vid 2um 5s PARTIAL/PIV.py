from openpiv import tools, pyprocess, validation , filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import imageio

frame_a = tools.imread('./data/DataSet1/657324.tif')
frame_b = tools.imread('./data/DataSet1/657325.tif')

# fig,ax = plt.subplots(1,2)
# img1 = ax[0].imshow(frame_a, cmap = plt.cm.gray)
# img2 = ax[1].imshow(frame_b, cmap = plt.cm.gray)

winsize = 16 # pixels, interrogation window size in frame A
searchsize = 28  # pixels, search in image B
overlap = 8 # pixels, 50% overlap
dt = 0.02 # sec, time interval between pulses


u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                       frame_b.astype(np.int32), 
                                                       window_size=winsize, 
                                                       overlap=overlap, 
                                                       dt=dt, 
                                                       search_area_size=searchsize, 
                                                       sig2noise_method='peak2peak')
# print(u0,v0)
# print(sig2noise)

x, y = pyprocess.get_coordinates( image_size=frame_a.shape, 
                                  search_area_size=searchsize, 
                                  overlap=overlap )

flags = validation.sig2noise_val( sig2noise, 
                                  threshold = 1.05 )

flags_g = validation.global_val( u0, v0, (-15, 15), (-15, 15) )
# flags = flags | flags_g

u2, v2 = filters.replace_outliers( u0, v0, 
                                   flags,
                                   method='localmean', 
                                   max_iter=3, 
                                   kernel_size=3)

# print(flags)

x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                               scaling_factor = 96.52 ) # 96.52 microns/pixel

x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

tools.save('Analysis.txt', x, y, u3, v3, flags)

fig, ax = plt.subplots()
tools.display_vector_field('Analysis.txt', 
                           ax=ax, scaling_factor=96.52, 
                           scale=20, # scale defines here the arrow length
                           width=0.0015, # width is the thickness of the arrow
                           on_img=True, # overlay on the image
                           image_name='./data/DataSet1/657324.tif');

plt.show()

