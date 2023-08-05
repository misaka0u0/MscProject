from openpiv import tools, pyprocess, validation , filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import imageio

winsize = 64 # pixels, interrogation window size in frame A 64
searchsize = 72  # pixels, search in image B 72
overlap = 12 # pixels, 50% overlap
dt = 0.02 # sec, time interval between pulses


objectA = np.load('stkA.npy')
objectB = np.load('stkB.npy') 

# objectA = np.load('deconvolved_RL1.npy')
# objectB = np.load('deconvolved_RL2.npy') 

velocity_stack = []
for i in range(objectA.shape[0]):

    img1 = objectA[i]
    img2 = objectB[i]

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(img1.astype(np.int32), 
                                                       img2.astype(np.int32), 
                                                       window_size=winsize, 
                                                       overlap=overlap, 
                                                       dt=dt, 
                                                       search_area_size=searchsize, 
                                                       sig2noise_method='peak2peak')

    flags = validation.sig2noise_val( sig2noise, 
                                  threshold = 1.05 )

    u0, v0 = filters.replace_outliers( u0, v0, 
                                       flags,
                                       method='localmean', 
                                       max_iter=3, 
                                       kernel_size=3)

    v0 = v0 / 96.52                    #scaling_factor = 96.52 # 96.52 microns/pixel


    velocity_stack.append(np.max(v0, axis=0))  # Using the maximum velocity value along y-axis for each layer

# Convert to a numpy array for convenience
velocity_stack = np.array(velocity_stack)

# Now you can create a heatmap. Note that we use the `imshow` function
# and also include a colorbar.
plt.figure(figsize=(10, 8))
plt.imshow(velocity_stack, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label='Velocity')
plt.xlabel('Y position')
plt.ylabel('Z position')
plt.title('Velocity Heatmap')
plt.show()



#     v0 = np.average(v0)
#     # v0 = np.max(v0)
#     velocity_stack.append(v0)

# # print(velocity_stack)

# x = range(objectA.shape[0])

# y = velocity_stack
# fig = plt.figure()
# ax  = fig.add_subplot(111)

# # Fit a parabola to the data
# coeffs = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
# poly = np.poly1d(coeffs)  # Create a polynomial function
# yfit = poly(x)  # Generate fitted y-values

# # Plot
# plt.scatter(x, y, label='velocity profile')
# plt.plot(x, yfit, color='red', label='fitted curve')  # Plot the fitted curve

# # plt.plot(x, y, label='velocity profile')
# plt.xlabel('z')
# plt.ylabel('velocity')
# plt.title('velocity profile')
# plt.legend()
# plt.grid(True)
# plt.ylim([-4, 12])
# plt.show()
