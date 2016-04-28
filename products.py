#!/usr/bin/env python
#coding=utf8

import os, sys
import numpy as np
import tables
from pyhdf import SD
from scipy.signal import medfilt
import netCDF4
from pyproj import Proj
from scipy.spatial import cKDTree
from glasslab_cluster.io import modis

KNN_PROC = 9

def fillinvalid(img, invalid=None, winsize=11, maxinvalid=0.35, pad=True, validrange=None, fillable=None):
    """Fill the invalid pixels in img indicated by either validrange
    or invalid. The filling is done by taking the average of the valid
    pixels in a 2D window to predict the center pixel.  The first
    window with at most maxinvalid fraction of invalid pixels out of
    sucessively large windows starting from shape (3,3) and ending
    with shape (winsize, winsize) is used. If no such window
    exists, the center pixel is not filled in.

    Parameters
    ----------
    img : 2d ndarray
        Image containing invalid pixels.
    invalid : 2d ndarray, optional
        Boolean mask of the same shape as img indicating
        the location of invalid pixels.
    winsize : int, optional
        Maximum size of window. Must be odd.
    maxinvalid : float, optional
        If there are less than maxinvalid fraction of invalid
        pixels in the window, the window is used for filling.
    pad : bool, optional
        Pad the image so that winsize//2 pixels at the
        border are also filled.
    validrange : 2-tuple, optional
        Values in the given range is considered valid.
    fillable : 2d ndarray, optional
        Boolean mask of the same shape as img where
        True indicates attempt will be made to fill the pixel
        and Flase indicates the pixel will not be filled.

    Returns
    -------
    new_img : 2d ndarray
        The filled image.
    new_invalid : 2d ndarray
        New invalid mask indicating which pixels were not filled.

    """
    if invalid is None:
        if len(validrange) != 2:
            raise ValueError("validrange has length %d" % (len(validrange),))
        invalid = (validrange[0] > img) | (img > validrange[1])
    if fillable is None:
        fillable = invalid.copy()
    if img.shape != invalid.shape:
        raise ValueError("img.shape %s != invalid.shape %s" % (img.shape, invalid.shape))
    if img.shape != fillable.shape:
        raise ValueError("img.shape %s != fillable.shape %s" % (img.shape, fillable.shape))
    if int(winsize)%2 != 1 or int(winsize) < 3:
        raise ValueError("winsize=%s must be an odd integer >= 3" % winsize)
    if 0 > maxinvalid or maxinvalid > 1:
        raise ValueError("maxinvalid=%s must be in the range [0,1]" % maxinvalid)

    invalid = np.array(invalid, dtype='bool')
    maxpad = min(int(winsize)//2, img.shape[0], img.shape[1])
    winsize = None  # winsize is wrong if img is smaller than winsize

    if pad:
        img = _mirrorpad(img, width=maxpad)
        invalid = _mirrorpad(invalid, width=maxpad)
        fillable = _mirrorpad(fillable, width=maxpad)
    newinvalid = invalid.copy()
    newimg = img.copy()

    # Don't try to fill pixels near the border
    fillable = fillable.copy()
    fillable[:maxpad,:], fillable[-maxpad:,:] = False, False
    fillable[:,:maxpad], fillable[:,-maxpad:] = False, False

    for i, j in zip(*np.where(fillable)):
        for p in xrange(1, maxpad+1):
            ind = np.s_[i-p:i+p+1, j-p:j+p+1]
            win = img[ind]
            wininv = invalid[ind]
            if np.sum(wininv)/(2.0*p+1.0)**2 <= maxinvalid:
                newimg[i,j] = win[~wininv].mean()
                newinvalid[i,j] = False
                break
    if pad:
        newimg = _mirrorunpad(newimg, width=maxpad)
        newinvalid = _mirrorunpad(newinvalid, width=maxpad)

    return newimg, newinvalid

def _get4widths(width=None, winsize=None):
    if width is None:
        if len(winsize) != 2:
            raise ValueError("winsize must be a 2-tuple")
        return (winsize[0]//2, winsize[0]//2, winsize[1]//2, winsize[1]//2)
    try:
        width = int(width)
        width = (width, width, width, width)
    except TypeError:
        width = tuple(width)
        if len(width) != 4:
            raise ValueError("width must be either an integer or a 4-tuple")
    if any([x < 0 for x in width]):
        raise ValueError("negative value in width=%s" % width)
    return width

def _mirrorpad(img, width=None, winsize=None):
    """Return the image resulting from padding width amount of pixels on
    each sides of the image img.  The padded values are mirror image with
    respect to the borders of img.

    Either width or winsize must be specified. Width can be an integer
    or a tuple (north, south, east, west). Winsize of (r, c) corresponds to
    a width of (r//2, r//2, c//2, c//2).

    """
    n, s, e, w = _get4widths(width=width, winsize=winsize)

    rows = []
    if n != 0:
        north = img[:n,:]
        rows.append(north[::-1,:])
    rows.append(img)
    if s != 0:
        south = img[-s:,:]
        rows.append(south[::-1,:])
    if len(rows) > 1:
        img = np.row_stack(rows)

    cols = []
    if w != 0:
        west = img[:,:w]
        cols.append(west[:,::-1])
    cols.append(img)
    if e != 0:
        east = img[:,-e:]
        cols.append(east[:,::-1])
    if len(cols) > 1:
        img = np.column_stack(cols)
    return img

def _mirrorunpad(img, width=None, winsize=None):
    """Return unpadded image of img padded with :func:`_mirrorpad`."""
    n, s, e, w = _get4widths(width=width, winsize=winsize)
    # index of -0 refers to the first element
    if s == 0:
        s = img.shape[0]
    else:
        s = -s
    if e == 0:
        e = img.shape[1]
    else:
        e = -e
    return img[n:s, w:e]

def _low2hi_swath(d):
    colpts = np.zeros((d.shape[0],2*d.shape[1]))
    colpts[:,0::2] = d
    colpts[:,1:-2:2] = (d[:,:-1]+d[:,1:])/2.0
    colpts[:,-1] = 2.0*colpts[:,-2]-colpts[:,-3]

    result = np.zeros((2*d.shape[0],2*d.shape[1]))
    result[1:-1:2,:] = (3.0*colpts[0:-1,:]/4.0)+(colpts[1:,:]/4.0)
    result[2:-1:2,:] = (colpts[0:-1,:]/4.0)+(3.0*colpts[1:,:]/4.0)
    result[0,:] = 2.0*colpts[0,:]-result[1,:]
    result[-1,:] = 2.0*colpts[-1,:]-result[-2,:]
    return result

def _interp_low2hi(lowres, swath_size=10):
    """Interpolate a MODIS image to double its resolution."""     
    return np.vstack([_low2hi_swath(swath) for swath in 
                     [lowres[i*swath_size:(i+1)*swath_size,:] for i in xrange(lowres.shape[0]/swath_size)]
                    ])

def interp_nasa(img, baddets, NDetectors=20):
    """Fill in lines from bad detectors by linear interpolation
    of neighboring lines from good detectors.
    Parameters
    ----------
    img : 2d ndarray
        Band image. Modified in-place.
    baddets : 1d ndarray
        Bad detectors.
    Returns
    -------
    img : 2d ndarray
        Input img modified in-place.
    """
    gooddets = [d for d in xrange(NDetectors) if d not in baddets]

    left = -2*NDetectors + np.zeros(NDetectors)
    right = -2*NDetectors + np.zeros(NDetectors)
    for b in baddets:
        # Assume there is always a detector on the left in the current scan line.
        # There may not exist a detector on the right (last detector on 500m
        # resolution is bad).
        gr = [g for g in gooddets if g > b]
        if len(gr) != 0:
            right[b] = min(gr)
        left[b] = max([g for g in gooddets if g < b])

    for i in xrange(img.shape[0]//NDetectors):
        for b in baddets:
            k = i*NDetectors+b
            if right[b] < 0:
                img[k, :] = img[i*NDetectors+left[b], :]
            else:
                mu = (b-left[b])/float(right[b]-left[b])
                img[k, :] = (1-mu)*img[i*NDetectors+left[b], :] + mu*img[i*NDetectors+right[b], :]
    return img

def _hdf_read(hdffile, var):
    sd = SD.SD(hdffile, SD.SDC.READ)
    v = sd.select(var)[:]
    sd.end()
    return v

class MODIS():
    """
    Initialize with MODIS file paths for M_D02HKM,M_D021KM and M_D03 files
    """
    def __init__(self, mod02hkm, mod021km, mod03, mod29=None): #, mod06):
        if not os.path.exists(mod02hkm):
            raise RuntimeError, "M_D02HKM file [ %s ] does not exist" % (mod02hkm)
        if not os.path.exists(mod021km):
            raise RuntimeError, "M_D021KM file [ %s ] does not exist" % (mod021km)
        if not os.path.exists(mod03):
            raise RuntimeError, "M_D03 file [ %s ] does not exist" % (mod03)
        if mod29 and not os.path.exists(mod29):
            raise RuntimeError, "M_D29 file [ %s ] does not exist" % (mod29)
        self.hkm = mod02hkm
        self.km  = mod021km
        self.geo = mod03
        self.mod29 = mod29

    @property
    def _sensor_zenith(self):
        sd = SD.SD(self.geo,SD.SDC.READ)
        ds = sd.select('SensorZenith')
        sz = np.deg2rad(ds[:]*getattr(ds,'scale_factor'))
        sd.end()
        return sz

    @property
    def _solar_zenith(self):
        sd = SD.SD(self.geo,SD.SDC.READ)
        ds = sd.select('SolarZenith')
        sz = np.deg2rad(ds[:]*getattr(ds,'scale_factor'))
        sd.end()
        return sz

    @property
    def sea_ice(self):
        sd = SD.SD(self.mod29,SD.SDC.READ)
        ds = sd.select('Sea_Ice_by_Reflectance')
        mask = np.array(ds[:], dtype='uint8')
        sd.end()
        return mask

    @property
    def rgb(self):

        ref = self.reflectance([1,4,3])
        return np.dstack(ref)

    @property
    def ist(self):

        A = np.array([[-1.5711228087,-2.3726968515,-4.2953046345],
                      [-0.1594802497,-3.3294560023,-5.2073604160]])
        B = np.array([[ 1.0054774067, 1.0086040702, 1.0150179031],
                      [ 0.9999256454, 1.0129459037, 1.0194285947]])
        C = np.array([[ 1.8532794923, 1.6948238801, 1.9495254583],
                      [ 1.3903881106, 1.2145725772, 1.5102495616]])
        D = np.array([[-0.7905176303,-0.2052523236, 0.1971325790],
                      [-0.4135749071, 0.1310171301, 0.2603553496]])

        def _get_GRingPointLatitude():
            sd = SD.SD(self.geo, SD.SDC.READ)
            meta = getattr(sd,'CoreMetadata.0')
            sd.end()
            mdl = meta.splitlines()
            in_obj = False
            for l in mdl:
                if l == '':
                    continue
                k,v = l.split('=',1)
                if k.strip()=="OBJECT" and v.strip()=="GRINGPOINTLATITUDE":
                    in_obj = True
                    continue
                if in_obj and k.strip() == "VALUE":
                    return np.array(map(float,v.strip()[1:-1].split(',')))

        GRingPointLatitude = _get_GRingPointLatitude()

        if (GRingPointLatitude.min()>0):
            hemi = 0
        elif(GRingPointLatitude.max()<0):
            hemi = 1
        else:
            if(GRingPointLatitude.max()>= -GRingPointLatitude.min()):
                hemi = 0
            else:
                hemi = 1

        hdf = modis.Level1B(self.km)
        t31 = hdf.temperature(31).read().data
        t32 = hdf.temperature(32).read().data

        constant_index = np.zeros(t31.shape,dtype='uint8')
        constant_index[(t31>=240)&(t31<=260)] = 1
        constant_index[(t31>260)] = 2

        return A[hemi,:][constant_index] + \
               B[hemi,:][constant_index] * t31 + \
               C[hemi,:][constant_index] * (t31-t32) + \
               D[hemi,:][constant_index] * ((t31-t32)*(1/np.cos(self._sensor_zenith)-1))


    def latlon(self,res='km'):
        lat = _hdf_read(self.geo,'Latitude')
        lon = _hdf_read(self.geo,'Longitude')
        
        if res=='km':
            return lat, lon
        if res=='hkm':
            return _interp_low2hi(lat,swath_size=10), \
                   _interp_low2hi(lon,swath_size=10)
        if res=='qkm':
            return _interp_low2hi( 
                     _interp_low2hi(lat,swath_size=10),
                     swath_size=20), \
                   _interp_low2hi(
                     _interp_low2hi(lon,swath_size=10), \
                     swath_size=20)

    def reflectance(self, bands, max_fill_itrs=10):
        bands = map(int, bands)

        hdf = modis.Level1B(self.hkm)
        sza = _interp_low2hi(self._solar_zenith, swath_size=10)
        sza_mask = (sza >= np.deg2rad(87))

        reflectance = []
        for b in bands:
            band = hdf.reflectance(b)
            bdata = band.read()
            data = bdata.data
            data[bdata.mask] = np.nan

            valid_range = band.valid_range()
            invalid = (data<valid_range[0])|(data>valid_range[1])
            
            if np.any(invalid):
                fill_mask = invalid & (~bdata.mask)
                fill_maxinv = 0.4
                fill_win = 11

                for i in range(max_fill_itrs):
                    data, fi = fillinvalid(data,fill_mask,
                                            winsize=fill_win,
                                            maxinvalid=fill_maxinv,
                                            validrange=valid_range)
                    if not np.any(fi):
                        break

                    if fill_maxinv < 0.9:
                        fill_maxinv+= 0.1
                        continue

                    fill_maxinv = 0.4
                    fill_win += 10
                else:
                    data[data>1.6] = 1.6

            data /= np.cos(sza)
            data[sza_mask] = np.nan

            #Fix broken detector in band 5 
            if b == 5:
                data = interp_nasa(data, hdf.dead_detectors()[str(b)])
            
            #destripe band
            if (self.hkm[-44:-41]=='MOD') and (b in [3,5,7]):
                d1 = band.destripe(data[:,0::2])
                d2 = band.destripe(data[:,1::2])
                data[:,0::2] = d1
                data[:,1::2] = d2
                reflectance.append((band.destripe(data.T)).T)
            else:
                reflectance.append(band.destripe(data))      

        return np.array(reflectance)

    def project(self, proj4_str, tx, ty, data, res='hkm', nn=4, md=1500):        
        p = Proj(proj4_str)

        lat, lon = self.latlon(res)
        sx, sy = p(lon,lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            raise ValueError, "no overlap between MODIS data and input grid"

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),k=nn,n_jobs=KNN_PROC)

        if nn==1:
            if len(data.shape) > 2:
                projected = []
                for band in data:
                    proj_band = band[m][ind]
                    proj_band[dist>md] = np.nan
                    projected.append(proj_band.reshape(tx.shape))
                return np.array(projected)
            else:
                proj_band = data[m][ind].astype('f8')
                proj_band[dist>md] = np.nan
                return proj_band.reshape(tx.shape)
        else:
            inv_dist = 1/dist
            normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]

            if len(data.shape) > 2:
                projected = []
                for band in data:
                    proj_band = band[m][ind]
                    proj_band[dist>md] = np.nan
                    projected.append(np.sum(proj_band*normed_dist,axis=1).reshape(tx.shape))
                return np.array(projected)
            else:
                proj_band = data[m][ind]
                proj_band[dist>md] = np.nan
                return np.sum(proj_band*normed_dist,axis=1).reshape(tx.shape)

