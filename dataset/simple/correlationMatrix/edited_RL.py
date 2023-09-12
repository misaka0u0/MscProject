from scipy.signal import convolve
import numpy as np

def richardson_lucy_edge(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be n-dimensional).
    psf : ndarray
       The point spread function.
    num_iter : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import img_as_float, data, restoration
    >>> camera = img_as_float(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> rng = np.random.default_rng()
    >>> camera += 0.1 * camera.std() * rng.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    #float_type = _supported_float_type(image.dtype)
    float_type = np.float64    # A quick hack to save me working out how to import _supported_float_type !
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    # Small regularization parameter used to avoid 0 divisions
    eps = 1e-12

    ones = np.ones_like(image)
    convOnes = convolve(ones, psf_mirror, mode='same') + eps
    
    for _ in range(num_iter):
        conv = convolve(im_deconv, psf, mode='same') + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= (convolve(relative_blur, psf_mirror, mode='same') / convOnes)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv
