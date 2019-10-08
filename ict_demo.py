#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:40:25 2019

@author: Julien FLEURET, julien.fleuret.1@ulaval.ca
"""

import numpy as np, h5py as h5, os, cv2
from scipy.io import loadmat
from scipy.stats import iqr
from scipy.signal import medfilt2d
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
from sklearn.decomposition import FastICA, PCA
from matplotlib import pyplot as plt


class openmat_h5_:
    
    def __init__(self, filename, what):
        
        assert os.path.exists(filename)
        
        self.fid = h5.File(filename, 'r')
        
        
        
        if not what is None:
            if isinstance(what,(tuple, list, np.ndarray)):
                if len(what) == 0:
                    self.use_acceptance = False
                    self.acceptance = None
                else:
                    self.use_acceptance = True
                    self.acceptance = what
            elif isinstance(what, (str,np.str)):
                self.use_acceptance = True
                self.acceptance = [what]
        else:
            self.use_acceptance = False
            self.acceptance = None   
            
        
    def _read_group(self, branch_id):
        
        ret = dict()
        
        for key in list(branch_id.keys()):
            
            try:            
                item = branch_id[key]
                            
                if isinstance(item, h5.Group):
                    ret[key] = self._read_group(item)
                elif isinstance(item, h5.Dataset):
                    ret[key] = item[...]
            except:
                continue
            
        return ret
        
    def _read_group_acceptance(self, branch_id):
        
        ret = dict()
        
        for key in list(branch_id.keys()):

            try:
                if key in self.acceptance:
                    item = branch_id[key]
                                
                    if isinstance(item, h5.Group):
                        ret[key] = self._read_group(item)
                    elif isinstance(item, h5.Dataset):
                        ret[key] = item[...]
            except:                
                continue
            
        return ret  
    
    
    def parse(self):
        
        ret = dict()
        
        if self.use_acceptance:
            for key in list(self.fid.keys()):
                
                if key in list(self.acceptance):
                    
                    item = self.fid[key]
                    
                    if isinstance(item, h5.Group):
                        ret[key] = self._read_group_acceptance(item)
                    else:
#                        ret += self._read_dataset_acceptance(item)
                        ret[key] = item[...]
        else:
            
            for key in list(self.fid.keys()):
                
#                print(key)
                
                item = self.fid[key]

                if isinstance(item, h5.Group):
                    ret[key] = self._read_group(item)
                else:
                    ret[key] = item[...]

        return ret

def openmat(filename, field = None):
    """ openmat(filename[, field ])->dst
    @brief Open a .mat file and from its content return it all (field == -1) or
    return the specified field or fields as specified to the variable field
    
    @param filename  Name of the file to be open.
    @param fields Name of the field, or fields to be loaded. 
    If fields is set to -1 all the field content in the file will be load.
    """
    tmp = []
    
    try:
        
        data = loadmat(filename)
        
        if isinstance(field,(str, np.str)):
            tmp = data[field]
        elif isinstance(field, (tuple, list)):
                        
            for key in data.keys():
                if '__header__' in key:
                    continue
                if '__version__' in key:
                    continue
                if '__globals__' in key:
                    continue
                
                if key in field:                
                    tmp.append(data[key])
        else:
            
            for key in data.keys():
                if '__header__' in key:
                    continue
                if '__version__' in key:
                    continue
                if '__globals__' in key:
                    continue
                
                tmp.append(data[key])            
    except:
        
        read_file = openmat_h5_(filename, field)
        
        tmp = read_file.parse()
        
    return tmp

def fromSequenceToOverlay(sequence, frames_first = False, apply_standardization = True):
    """ fromSequenceToOverlay(sequence [, frame_first [, apply_normalization] ]) -> overlay, dims
    @brief This function return the overlay matrix of a sequence of images.
    The overlay is the a matrix in which each column is the flatten
    representation of an image of the input sequence.
    
    @param sequence It can be a 3d ndarray, a list or a tuple of ndarray.
    @param frames_first If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    @param apply_normalization If true the overlay will be centred and
    reduced by its variance.    
    """    
    sequence = np.asarray(sequence)
    
    assert sequence.ndim == 3
    
    if frames_first:
        frames, rows, cols = sequence.shape
        sequence = sequence.reshape((-1, rows * cols))
    else:
        rows, cols, frames = sequence.shape
        sequence = sequence.reshape((rows * cols, -1)).T
    
    if apply_standardization:
        
        mu = np.mean(sequence, axis=0)
        sigma = np.std(sequence, axis=0)
        
        sigma[sigma==0] = 1
                
        sequence = (sequence - mu) / sigma
    
    return sequence, (rows, cols, frames)

def fromOverlayToSequence(overlay, dims, frames_first = False):
    
    """ fromOverlayToSequence(overlay, dims [, frames_first])->sequence
    @brief  This function return a sequence of frames from a overlay matrix.
    @param overlay a 2D matrix (ndarray). Each column correspond to an image.
    @param dims a list, a tuple or a ndarray containing the number of rows,
    cols and frames.
    If dims is set to None it means that the overlay must be return as is.
    @param frames_first : should the returned sequence be organize with the 
    frames as first dimension ? (default: False)
    @param sequence a 3D matrix (ndarray) 
    """
    
    assert isinstance(dims,(list, tuple, np.ndarray)) or dims is None
    
    if dims is None:
        return overlay
            
    rows, cols, frames = dims
    
    overlay = overlay.reshape((-1, rows, cols))
    
    if overlay.shape[0] > frames:
        overlay = overlay[:frames]
    
    if not frames_first:
        overlay = overlay.transpose((1,2,0))
    
    return overlay

def ApplyNCT(method, sequence, apply_standardization, frames_first, **kwargs):
    """ ApplyNCT(method, sequence, apply_normalization, frames_first, **kwargs)-> method_argouts
    @brief This function apply a sequence 
    
    @param method function or object that will be used to process the data.
    
    @param sequence either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param apply_standardization boolean value to set in order to apply a standardization or not
    
    @param frame_first If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.

    @param method_argouts output argmuments resulting from the processing of the variable sequence by the variable assign in method
    
    """
    sequence = np.asarray(sequence)
    
    assert sequence.ndim == 3 or sequence.ndim == 2
    
    reconstruct_before_return = sequence.ndim == 3

    
    nct = method(**kwargs)

    
    overlay, (rows, cols, frames) = fromSequenceToOverlay(sequence,
                         frames_first=frames_first,
                         apply_standardization=apply_standardization)
    
    overlay_pct = nct.fit(overlay).components_
    
    return fromOverlayToSequence(overlay_pct, \
                                 (rows, cols, frames) if reconstruct_before_return else None, \
                                 frames_first=frames_first)

def PCT(sequence, n_components = None, apply_standardization = True, frames_first = False):
    """PCT(sequence [,n_components [, apply_normalization [, frames_first] ] ])
    -> sequence_pct
    @brief This function implement the Principal Component Thermograpy as describe by
    Rajic.
    
    @param sequence either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param n_components the number of component to keep. If not set the 
    sequence returned as the same size a the input sequence.
    
    @param apply_normalization if set the overlay matrix on which the PCT is
    processed will first be centred around its mean and reduce by its variance.
    
    @param frame_first If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param sequence_pct Sequence reconstructed after application of PCT.
    """

    return ApplyNCT(PCA, sequence, frames_first=frames_first, \
                    apply_standardization=apply_standardization, \
                    n_components=n_components)

def ICT(sequence, n_components = None, apply_standardization = True, frames_first = False, **kwargs):
    """PCT(sequence [,n_components [, apply_normalization [, frames_first
    [, **kwargs] ] ] ]) -> sequence_pct
    @brief This function implement the Independant Component Thermograpy as describe
    by no one yet.
    
    @param sequence either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param n_components the number of component to keep. If not set the 
    sequence returned as the same size a the input sequence.
    
    @param apply_normalization : if set the overlay matrix on which the PCT is
    processed will first be centred around its mean and reduce by its variance.
    
    @param frame_first If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param kwargs this code use the function FastICA from scikit-learn.
    Kwargs allow to send some extra configuration parameters to the function
    FastICA.
    
    @param sequence_pct : sequence reconstructed after application of PCT.
    """

    return ApplyNCT(FastICA, sequence, frames_first=frames_first, \
                    apply_standardization=apply_standardization, \
                    n_components=n_components, **kwargs)

def get_roi(I, frames_first=False):
    """get_roi(I [, frames_first]) -> roi
    @brief This function look for a rectangular region of interest in an image and return its bounding box parameters.
    
    @param I source image or sequence.
    
    @param frame_first If true it means that the parameter I is a sequence from which
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols.
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param roi main parameters of the roi found in the image.
    i.e. coordinates (x,y) and size (width, height)
    
    """
    if I.ndim == 3:
        
        frames_first = False
            
        I = np.squeeze(I[0,:,:]) if frames_first else np.squeeze(I[:,:,0])
                    
            
    
    if not np.issubdtype(I.dtype, np.uint8):
        
        I = I.astype(np.float32)
        
        mu, sigma = cv2.meanStdDev(I)
        
        I = (I-mu)/sigma
        
        I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    
    _, It = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)

    It = remove_small_holes(remove_small_objects(It.astype(np.bool) ) ).astype(np.uint8) * 255
    
    
#   Rows analysis    
        
    rows, cols = It.shape
    Ir = np.zeros_like(It)
    
    cnz_r = np.zeros((rows,), np.int32)
    
    for r in range(rows):
            cnz_r[r] = np.count_nonzero(It[r])        

    for r in range(rows):
        cnz_r[r] = np.count_nonzero(It[r])        

    mu = cnz_r.mean()

    for r in range(rows):
        if cnz_r[r] >= mu:
            Ir[r] = It[r]
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(Ir)

    areas = stats[1:, -1]
    
    idx = np.argmax(areas)+1
    
    x, y, w, h = stats[idx,:-1]

    Ir = np.zeros_like(It)

    Ir[y:y+h, x:x+w] = It[y:y+h, x:x+w]

#   Columns analysis    

    cnz_c = np.zeros((cols,), np.int32)
    
    for c in range(cols):
            cnz_c[c] = np.count_nonzero(It[:, c])

    for c in range(cols):
        cnz_c[c] = np.count_nonzero(It[:, c])        

    mu = cnz_c.mean()

    for c in range(cols):
        if cnz_c[c] < mu:
            Ir[:,c] = 0
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(Ir)

    areas = stats[1:, -1]
    
    idx = np.argmax(areas)+1
    
    x, y, w, h = stats[idx,:-1]

    
    
    return x, y, w, h

class JensenShanon_:
    
    def __init__(self):
#        components_ : convinient variable name which allow to use ApplyNCT algorithm.
        self.entropy = None
        self.cst = np.log(2)

    def _computeEntropy(self, P, Q):
        
        PP, QQ = P.copy(), Q.copy()
        PP[PP<1] = 1
        QQ[QQ<1] = 1
        
        PQ = P+Q
        PQ[PQ<1] = 1
        
        left = P * ( (self.cst + np.log(PP) ) - np.log(PQ) )
        right = Q * ( (self.cst + np.log(QQ) ) - np.log(PQ) )
        
        return 0.5 * (np.sum(left) + np.sum(right) )

    def fit(self, sequence):
        
        sequence = np.asarray(sequence)
        
        frames, _ = sequence.shape
        
        self.entropy = np.zeros((frames,), np.float32)    
        
#        tmp = iqr(sequence, axis=0)        
#        
#        self.entropy[0] = self._computeEntropy(tmp, sequence[0])
        
        for f in range(frames-2):
            I_current = sequence[f]
            I_next = sequence[f+1]
            
            self.entropy[f+1] = self._computeEntropy(I_current, I_next)
        
        return self
        
def computeJensenShanon(sequence, frames_first):
    """computeJensenShanon(sequence, frames_first)->entropy
    @brief This function compute the divergence using the Jensen Shanon formulation.
    
    @param sequence either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param frame_first If true it means that the parameter I is a sequence from which
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols.
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.    

    @return entropy a vector of the same length of the sequence,
    which contains the interframe entropy of the images of the input sequence.
    Note in order to have the same number of elements than the number image,
    the first entropy is computed between the inter-quantile range of the sequence
    and the first frame.
    
    
    """
    
    sequence = np.asarray(sequence)
    
    assert sequence.ndim == 3 or sequence.ndim == 2
    
    
    JS = JensenShanon_()
    
    overlay, _ = fromSequenceToOverlay(sequence,
                         frames_first=frames_first,
                         apply_standardization=True)
            
    return JS.fit(overlay).entropy
    
    

if __name__ == '__main__':
        
    data = openmat('./data/CFRP_006_compressed.mat')
    if isinstance(data, dict):
        data = data['a65']
    else:
#       0, 1, 2 -> FLIR a65, FLIR phoenix, JenOptik Variocam HD
        data = data[0]

    use_automatic_finding = False    

    
    if isinstance(data, list):
        data = data[0]
        
        data = data.transpose((2,0,1))
    
    I8 = data[0].T
    mu, sigma = cv2.meanStdDev(I8)
    I8 = (I8 - mu) / sigma
#       The median filter is used in order to remove the dead pixels.
    I8 = medfilt2d(I8) 
    I8 = rescale_intensity(I8, out_range=(0,255)).astype(np.uint8)

    if not use_automatic_finding:        
        x,y,w,h = cv2.selectROI(windowName='please select the region of interest.', img=I8)        
        cv2.destroyWindow('please select the region of interest.')
    else:
        x,y,w,h = get_roi(I8)

    del I8    
#   Because the image has been transposed x,y,w,h must be reassign    
    x,y,w,h = y,x,h,w    
    
#   Compute the entropy, the moment where the flash have been activate correspond to the moment with the maximum entropy.    
    entropy = computeJensenShanon(data[:, y:y+h, x:x+w], frames_first=True)
    
    plt.figure()
    plt.plot(entropy)

#   From the entopy determine the moment where the flash have been activate, compute the cool image and reduce the dataset.    
    argmax = entropy.argmax()

    cool_image = np.mean(data[:argmax, y:y+h, x:x+w],axis=0)
            
    data = data[argmax:argmax+200, y:y+h, x:x+w] - cool_image

#   Compute both the ICT, and PCT, keep only the last 7 components.    
    data_ict = ICT(data, n_components=7, frames_first=True)
    data_pct = PCT(data, n_components=7, frames_first=True)
    
    for (I_ict, I_pct) in zip(data_ict, data_pct):
        
#       Apply a standization in order to ease the detection of the defects.        
        (mu_i, sigma_i), (mu_p, sigma_p) = cv2.meanStdDev(I_ict), cv2.meanStdDev(I_pct)
        
#       The abolute value is used in order not to care about the gradient inversion.
        I_ict, I_pct = np.abs( (I_ict - mu_i) / sigma_i), np.abs( (I_pct - mu_p) / sigma_p)
        
        I_ict, I_pct = I_ict >= threshold_otsu(I_ict), I_pct >= threshold_otsu(I_pct)
        
        I_ict, I_pct = remove_small_holes(remove_small_objects(I_ict)), remove_small_holes(remove_small_objects(I_pct))
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(I_ict, cmap='gray')
        plt.subplot(122)
        plt.imshow(I_pct, cmap='gray')
        plt.show()
        
