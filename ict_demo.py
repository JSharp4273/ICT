#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:40:25 2019

@author: Julien FLEURET, julien.fleuret.1@ulaval.ca
"""

import numpy as np, h5py as h5, os, cv2
from scipy.io import loadmat
from skimage.morphology import remove_small_holes, remove_small_objects
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
    
    @param filename : Name of the file to be open.
    @param fields : Name of the field, or fields to be loaded. 
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
    This function return the overlay matrix of a sequence of images.
    The overlay is the a matrix in which each column is the flatten
    representation of an image of the input sequence.
    
    @param sequence : It can be a 3d ndarray, a list or a tuple of ndarray.
    @param frames_first : If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    @param apply_normalization : If true the overlay will be centred and
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
    @param overlay : a 2D matrix (ndarray). Each column correspond to an image.
    @param dims : a list, a tuple or a ndarray containing the number of rows,
    cols and frames.
    If dims is set to None it means that the overlay must be return as is.
    @param frames_first : should the returned sequence be organize with the 
    frames as first dimension ? (default: False)
    @param sequence : a 3D matrix (ndarray) 
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
    This function apply a sequence 
    
    @param method : function or object that will be used to process the data.
    
    @param sequence : either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param apply_standardization : boolean value to set in order to apply a standardization or not
    
    @param frame_first : If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.

    @param method_argouts : output argmuments resulting from the processing of the variable sequence by the variable assign in method
    
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
    This function implement the Principal Component Thermograpy as describe by
    Rajic.
    
    @param sequence : either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param n_components : the number of component to keep. If not set the 
    sequence returned as the same size a the input sequence.
    
    @param apply_normalization : if set the overlay matrix on which the PCT is
    processed will first be centred around its mean and reduce by its variance.
    
    @param frame_first : If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param sequence_pct : sequence reconstructed after application of PCT.
    """

    return ApplyNCT(PCA, sequence, frames_first=frames_first, \
                    apply_standardization=apply_standardization, \
                    n_components=n_components)

def ICT(sequence, n_components = None, apply_standardization = True, frames_first = False, **kwargs):
    """PCT(sequence [,n_components [, apply_normalization [, frames_first
    [, **kwargs] ] ] ]) -> sequence_pct
    This function implement the Independant Component Thermograpy as describe
    by no one yet.
    
    @param sequence : either a overlay matrix, a three dimensional array, or a tuple or
    a list of matrix.
    
    @param n_components : the number of component to keep. If not set the 
    sequence returned as the same size a the input sequence.
    
    @param apply_normalization : if set the overlay matrix on which the PCT is
    processed will first be centred around its mean and reduce by its variance.
    
    @param frame_first : If true it means that the parameter sequence 
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param kwargs : this code use the function FastICA from scikit-learn.
    Kwargs allow to send some extra configuration parameters to the function
    FastICA.
    
    @param sequence_pct : sequence reconstructed after application of PCT.
    """

    return ApplyNCT(FastICA, sequence, frames_first=frames_first, \
                    apply_standardization=apply_standardization, \
                    n_components=n_components, **kwargs)

def get_roi(I, frames_first=False):
    """get_roi(I [, frames_first]) -> roi
    This function look for a rectangular region of interest in an image and return its bounding box parameters.
    
    @param I : source image or sequence.
    
    @param frame_first : If true it means that the parameter I is a sequence from which
    dimensions are organize in way that the number of frames is represented by
    the first dimension.
    Oftenly this organization shape is: frames, rows, cols.
    If frame_first is False the shape of the data is supposed to be organized
    like this: rows, cols, frames.
    
    @param roi : main parameters of the roi found in the image.
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

if __name__ == '__main__':
    
    fn1 = '/media/smile/New Volume/julien/materials/cfrp_006_front.mat'
    fn2 = '/home/smile/octave_dir/joint_experiment_lei_lei/CFRP_Ctrl.mat'
    
    data = openmat(fn2)
#    data = data['a65']
    
    if isinstance(data, list):
        data = data[0]
        
        data = data.transpose((2,0,1))
    
#    data = np.asarray([d.T for d in data])
    
    x,y,w,h = get_roi(data[0])
    
    data = data[:400, y:y+h, x:x+w]
    
    data_ict = ICT(data, n_components=7, frames_first=True)
    data_pct = PCT(data, n_components=7, frames_first=True)

    
    for (I_ict, I_pct) in zip(data_ict, data_pct):
        plt.figure()
        plt.subplot(121)
        plt.imshow(I_ict, cmap='gray')
        plt.subplot(122)
        plt.imshow(I_pct, cmap='gray')

