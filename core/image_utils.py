import numpy as np
import math
import scipy
from skimage import measure
from scipy import ndimage
from matplotlib import path
# import os
def contour_main_obj(im):
	contours = measure.find_contours(im,0.5)
	idx = 0
	con = contours[0]
	length = len(con)
	for n,contour in enumerate(contours):
		k = len(contour)
		if k>length:
			idx = n
			length = k
			con = contours[n]
	return con

def downsample_con(con,R):
    x = np.squeeze(con[:,1])
    y = np.squeeze(con[:,0])
    crop_x = x.size-math.floor(float(x.size)/R)*R
    x_crop = x[0:(len(x)-crop_x)]
    x_R = x_crop.reshape(-1,R).mean(axis=1)
    x_R = np.concatenate((x_R,[x_R[0]]))
    crop_y = y.size-math.floor(float(y.size)/R)*R
    y_crop = y[0:(len(y)-crop_y)]
    y_R = y_crop.reshape(-1,R).mean(axis=1)
    y_R = np.concatenate((y_R,[y_R[0]]))
    return np.stack((y_R,x_R),axis = -1)

def corner_cutting(coords,refinements = 5):
    coords =np.array(coords)
    for _ in range(refinements):
        L = coords.repeat(2,axis = 0)
        R = np.empty_like(L)
        R[0]  = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L*0.75 + R*0.25
    return coords

def localizer(bn_mask):
	contour = contour_main_obj(bn_mask)
	# R = len(contour)//16
	con_ds = downsample_con(contour,5)
	con_smooth = corner_cutting(con_ds,3)
	closed_path = path.Path(con_smooth)
	idx = np.array([[(j,i) for i in range(bn_mask.shape[0])] for j in range(bn_mask.shape[1])]).reshape(np.prod(bn_mask.shape),2)
	mask_smooth = closed_path.contains_points(idx).reshape(bn_mask.shape).astype(int)
	mask_dil = ndimage.binary_dilation(mask_smooth,iterations=4).astype(int)
	return mask_dil


def hist_match_grey(source, template):
	"""
	Adjust the pixel values of a grayscale image such that its histogram
	matches that of a target image
	Arguments:
	-----------
		source: np.ndarray
			Image to transform; the histogram is computed over the flattened
			array
		template: np.ndarray
			Template image; can have different dimensions to source
	Returns:
	-----------
		matched: np.ndarray
			The transformed output image
	"""
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)

def hist_match(source, template):
	outputs = np.zeros(source.shape)
	for i in range(source.shape[0]):
		for j in range(source.shape[1]):
			outputs[i, j] = hist_match_grey(source[i, j], template[i, j])

	return outputs
