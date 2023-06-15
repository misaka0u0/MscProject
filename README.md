<header>

  # Msc Project
_A backup, record and virsion control of my msc project_
  
</header>

<!--
This is comments block in GitHub markdown

-->

## Secion1. review 
> review on background, methodology, coding tool for realization.
[basic of the piv algorithms](https://openpiv.readthedocs.io/en/latest/src/piv_basics.html)

### PIV: what is it about

From Wikipedia: “Particle image velocimetry (PIV) is an optical method of flow visualization used in education and research. It is used to obtain instantaneous velocity measurements and related properties in fluids. The fluid is seeded with tracer particles which, for **sufficiently small particles, are assumed to faithfully follow the flow dynamics (the degree to which the particles faithfully follow the flow is represented by the *[Stokes number](https://en.wikipedia.org/wiki/Stokes_number)*).** The fluid with entrained particles is illuminated so that particles are visible. The motion of the seeding particles is used to calculate speed and direction (the velocity field) of the flow being studied.” Read more at http://en.wikipedia.org/wiki/Particle_image_velocimetry.


### code for tutorial in OpenPIV 

1. Keep the plot in notebook <br/>

`%matplotlib inline`is a magic function with % ahead, can used to keep the plot <br/><br/>


2. Read the image in .bmp(bitmap digital images) <br/>

```
frame_a  = tools.imread( path / "exp1_001_a.bmp" ) 
frame_b  = tools.imread( path / "exp1_001_b.bmp" )
``` 

#imread, result in 2Darray and seems dtype is uint8 as default<br/>

```
frame_a = frame_a.astype(np.int32) 
frame_b = frame_b.astype(np.int32) 
```
#frame_a.dtype dtype('int32') .astype to convert datatype<br/><br/>

3. Color map and color bar <br/>
  >color maps are used to visually represent the data<br/>

color maps can be found in some built-in lib such as [matplotlib.colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html). Or we can also [create custom color maps](https://matplotlib.org/stable/gallery/color/custom_cmap.html#sphx-glr-gallery-color-custom-cmap-py). <br/>

**The best colormap for any given data set depends on many things including:**

- Whether representing form or metric data ([Ware](http://ccom.unh.edu/sites/default/files/publications/Ware_1988_CGA_Color_sequences_univariate_maps.pdf))

- Your knowledge of the data set (e.g., is there a critical value from which the other values deviate?)

- If there is an intuitive color scheme for the parameter you are plotting

- If there is a standard in the field the audience may be expecting

>Combined with the colormaps, a color bar is essential to make it nice
```{
fig,ax = plt.subplots(1,2,figsize=(12,10))
im1 = ax[0].imshow(frame_a,cmap=plt.cm.gray)
im2 = ax[1].imshow(frame_b,cmap=plt.cm.gray)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im2, ax=ax, cax =cax)
}
```
## Section2. Methodology and improvement

## Section3. Results 

## Section4. Comments
