import numpy as np
from skimage.metrics import structural_similarity as ssim
# import cv2
# TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

def _error(pred, true):
    return pred - true

def _abs_error(pred, true):
    return np.absolute(_error(pred, true))

def _relative_error(pred, true):
    return _abs_error(pred, true) / true

def mean_relative_error(pred, true):
    return np.mean(_relative_error(pred, true))

def mean_squared_error(pred, true):
    return np.mean(np.square(_error(pred, true)))

def root_mean_squared_error(pred, true):
    return np.sqrt(mean_squared_error(pred, true))

def mean_log10_error(pred, true):
    return np.nanmean(_abs_error(np.log10(pred), np.log10(true)))

# def temporal_change_consistency(pred, true):
#     n = len(pred)
#     sum = 0
#     for i in range(n-1):
#         sum += ssim(np.absolute(pred[i]-pred[i+1]), np.absolute(true[i]-true[i+1]))
#     return sum / (n-1)

# def temporal_motion_consistency(pred, true):
#     n = len(pred)
#     sum = 0
#     for i in range(n-1):
#         sum += ssim(TVL1.calc(pred[i],pred[i+1],None), TVL1.calc(true[i],true[i+1],None), multichannel=True)
#     return sum / (n-1)

def bad_pix_ratio(pred, true, thresh=0.07):
    bad_pix = np.where(_abs_error(pred, true) > thresh, 1, 0)
    return np.sum(bad_pix)/(np.prod(bad_pix.shape)) * 100

def calc_metrics(pred=None, true=None):
    metrics = {}
    metrics['mre'] = mean_relative_error(pred, true)          if pred is not None else 0
    metrics['mse'] = mean_squared_error(pred, true)           if pred is not None else 0
    metrics['rmse'] = root_mean_squared_error(pred, true)     if pred is not None else 0
    metrics['log10'] = mean_log10_error(pred, true)           if pred is not None else 0
    # metrics['tcc'] = temporal_change_consistency(pred, true)  if pred is not None else 0
    # metrics['tmc'] = temporal_motion_consistency(pred, true)  if pred is not None else 0
    metrics['Badpix7'] = bad_pix_ratio(pred, true, thresh=0.07)    if pred is not None else 0
    metrics['Badpix3'] = bad_pix_ratio(pred, true, thresh=0.03)    if pred is not None else 0
    metrics['Badpix1'] = bad_pix_ratio(pred, true, thresh=0.01)    if pred is not None else 0
    return metrics