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
import pygrib

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

class BowTie(object):
    def __init__(self, lat):
        self.sortind = np.argsort(lat, axis=0)
        if lat[0,0] > lat[-1,0]:
            self.sortind = self.sortind[::-1,:]

        self.slat = self.sort(lat)

    def sort(self, img):
        assert img.ndim == 2
        return img[self.sortind, np.arange(img.shape[1])]

    def _interp(self, Y, valid, invalid):
        mask = (np.sum(valid, axis=0) >= 2) & \
            np.any(invalid, axis=0)
        for j in np.where(mask)[0]:
            vind, = np.where(valid[:,j])
            iind, = np.where(invalid[:,j])
            u, ui = np.unique(self.slat[vind,j], return_index=True)
            Y[iind,j] = np.interp(self.slat[iind,j], u, Y[vind[ui],j])
        return Y

    def fixfloat64(self, img):
        simg = self.sort(img)
        mimg = medfilt(simg, [3,1])
        newimg = np.copy(img)
        newimg[self.sortind, np.arange(mimg.shape[1])] = mimg

        mask = ~np.isfinite(img)
        newimg[mask] = img[mask]

        Y = self.sort(newimg)
        valid = np.isfinite(Y)
        invalid = np.isnan(Y)
        return self._interp(Y, valid, invalid)

    def unsort(self, img):
        u = np.zeros_like(img)
        u[self.sortind, np.arange(u.shape[1])] = img
        return u

class VIIRS_SDR(object):
    """
    Initialize with a dictionary of file paths,
    with data type identifiers as keys and lists
    of path stirngs as values.
    """
    def __init__(self, paths):
        def _check_paths(k, ps):
            for p in ps:
                if not os.path.exists(p):
                    raise RuntimeError, "%s file [ %s ] does not exist" % (k,p)
            return sorted(ps)

        #Geo Files
        self.GMTCO = _check_paths('GMTCO', paths['GMTCO']) if paths.has_key('GMTCO') else None
        self.GITCO = _check_paths('GITCO', paths['GITCO']) if paths.has_key('GITCO') else None
        #Imagery Files
        self.IMG = dict()
        self.IMG['SVI01'] = _check_paths('SVI01', paths['SVI01']) if paths.has_key('SVI01') else None
        self.IMG['SVI02'] = _check_paths('SVI02', paths['SVI02']) if paths.has_key('SVI02') else None
        self.IMG['SVI03'] = _check_paths('SVI03', paths['SVI03']) if paths.has_key('SVI03') else None
        self.IMG['SVI04'] = _check_paths('SVI04', paths['SVI04']) if paths.has_key('SVI04') else None
        self.IMG['SVI05'] = _check_paths('SVI05', paths['SVI05']) if paths.has_key('SVI05') else None
        #Moderate Files
        self.MOD = dict()
        self.MOD['SVM01'] = _check_paths('SVM01', paths['SVM01']) if paths.has_key('SVM01') else None
        self.MOD['SVM02'] = _check_paths('SVM02', paths['SVM02']) if paths.has_key('SVM02') else None
        self.MOD['SVM03'] = _check_paths('SVM03', paths['SVM03']) if paths.has_key('SVM03') else None
        self.MOD['SVM04'] = _check_paths('SVM04', paths['SVM04']) if paths.has_key('SVM04') else None
        self.MOD['SVM05'] = _check_paths('SVM05', paths['SVM05']) if paths.has_key('SVM05') else None
        self.MOD['SVM06'] = _check_paths('SVM06', paths['SVM06']) if paths.has_key('SVM06') else None
        self.MOD['SVM07'] = _check_paths('SVM07', paths['SVM07']) if paths.has_key('SVM07') else None
        self.MOD['SVM08'] = _check_paths('SVM08', paths['SVM08']) if paths.has_key('SVM08') else None
        self.MOD['SVM09'] = _check_paths('SVM09', paths['SVM09']) if paths.has_key('SVM09') else None
        self.MOD['SVM10'] = _check_paths('SVM10', paths['SVM10']) if paths.has_key('SVM10') else None
        self.MOD['SVM11'] = _check_paths('SVM11', paths['SVM11']) if paths.has_key('SVM11') else None
        self.MOD['SVM12'] = _check_paths('SVM12', paths['SVM12']) if paths.has_key('SVM12') else None
        self.MOD['SVM13'] = _check_paths('SVM13', paths['SVM13']) if paths.has_key('SVM13') else None
        self.MOD['SVM14'] = _check_paths('SVM14', paths['SVM14']) if paths.has_key('SVM14') else None
        self.MOD['SVM15'] = _check_paths('SVM15', paths['SVM15']) if paths.has_key('SVM15') else None
        self.MOD['SVM16'] = _check_paths('SVM16', paths['SVM16']) if paths.has_key('SVM16') else None
        #Intermediate Products
        self.IP = dict()
        self.IP['IVIQF'] = _check_paths('IVIQF', paths['IVIQF']) if paths.has_key('IVIQF') else None
        self.IP['IVIIC'] = _check_paths('IVIIC', paths['IVIIC']) if paths.has_key('IVIIC') else None
    
    @property
    def sic(self):
        sic = list()
        for path in self.IP['IVIIC']:
            h5 = tables.open_file(path,mode='r')
            sic.append(h5.get_node('/All_Data/VIIRS-I-Conc-IP_All/iceFraction').read())
            h5.close()

        return np.vstack(sic)
    

    def latlon(self,resolution):
        """
        Return latitude and longitude at specified resolution
        """
        lat = []
        lon = []
        
        if resolution=='M': 
            if not self.GMTCO:
                raise RuntimeError, "No GMTCO files provided."

            for path in self.GMTCO:
                h5 = tables.open_file(path,mode='r')
                lat.append(h5.get_node('/All_Data//VIIRS-MOD-GEO-TC_All/Latitude').read())
                lon.append(h5.get_node('/All_Data//VIIRS-MOD-GEO-TC_All/Longitude').read())
                h5.close()

        elif resolution=='I':
            if not self.GITCO:
                raise RuntimeError, "No GITCO files provided."

            for path in self.GITCO:
                h5 = tables.open_file(path,mode='r')
                lat.append(h5.get_node('/All_Data//VIIRS-IMG-GEO-TC_All/Latitude').read())
                lon.append(h5.get_node('/All_Data//VIIRS-IMG-GEO-TC_All/Longitude').read())
                h5.close()

        return np.vstack(lat), np.vstack(lon)

    def reflectance(self,band):
        """
        Return solar zenith corrected reflectances.
        Note: FillValue pixels set to NaN.
        """
        ref = []
        lat, _ = self.latlon(band[2])
        path_list = self.MOD[band] if band[2] == 'M' else self.IMG[band]

        for path in path_list:
            bid = band[2:5:2] if band[3] == '0' else band[2:5]
            h5 = tables.open_file(path, mode='r')
            raw = h5.get_node('/All_Data/VIIRS-%s-SDR_All/Reflectance'%bid).read().astype('f8')
            scale, offset = h5.get_node('/All_Data/VIIRS-%s-SDR_All/ReflectanceFactors'%bid).read()[:2]
            del_mask = (raw == 65533)
            night_mask = (raw == 65535)
            tref = (raw-offset)*scale
            tref[del_mask] = np.nan
            tref[night_mask] = -1            
            ref.append(tref)
            h5.close()
        
        return np.vstack(ref)

    def get_mask(self,kind='land'):
        """
        Return land from VIIRs fractional ice quality flags file.
        """
        MaskLand = 0x80
        MaskCloud = 0x18

        mask = []
        path_list = self.IP['IVIQF']
        for path in path_list:
            h5 = tables.open_file(path,mode='r')
            qf = h5.get_node('/All_Data//VIIRS-I-Qual-Flags-IP_All/QF2_VIIRSIceQualityIP').read() if kind=='land' else h5.get_node('/All_Data//VIIRS-I-Qual-Flags-IP_All/QF1_VIIRSIceQualityIP').read()
            mask.append(qf)
            h5.close()
        return ((np.vstack(mask)&MaskLand)>>7)>0 if kind=='land' else (np.vstack(mask)&MaskCloud)>>3

    def project(self, proj4_str, tx, ty, data, res, nn=4):
        md = 1000 if res == 'I' else 2000
        p = Proj(proj4_str)

        del_mask = np.any(np.isnan(data),axis=0)
        night_mask = np.any((data==-1)>0,axis=0)

        lat, lon = self.latlon(res)
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())&(~del_mask)
        
        if np.sum(m) < 100:
            raise ValueError, "no overlap between VIIRS data and input grid"          

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
                    proj_band[dist > md] = np.nan
                    proj_band[proj_band<0] = np.nan
                    projected.append(np.nansum(proj_band*normed_dist,axis=1).reshape(tx.shape))
                return np.array(projected)
            else:
                proj_band = data[m][ind]
                proj_band[dist > md] = np.nan
                proj_band[proj_band<0] = np.nan
                return np.nansum(proj_band*normed_dist,axis=1).reshape(tx.shape)

class ACSPO(object):
    def __init__(self, paths):
        for path in paths:
            if not os.path.exists(path):
                raise RuntimeError, "ACSPO file [ %s ] does not exist" % (path)
        self.paths = sorted(paths)

    def latlon(self):
        lat = []
        lon = []
        for path in self.paths:
            ds = netCDF4.Dataset(path, mode='r')
            lat.append(np.array(ds.variables['lat']))
            lon.append(np.array(ds.variables['lon']))
            ds.close()
        return np.vstack(lat), np.vstack(lon)

    def sst(self):
        ts = []
        for path in self.paths:
            ds = netCDF4.Dataset(path, mode='r')
            t = np.array(ds.variables['sea_surface_temperature']).astype('f8')
            t[t<=0] = np.nan
            bt = BowTie(np.array(ds.variables['lat']))
            ds.close()
            ts.append(bt.unsort(bt.fixfloat64(np.squeeze(t))))
        return np.vstack(ts)

    def project(self,proj4_str,tx,ty,nn=4,md=1500):
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            raise ValueError, "no overlap between ACSPO data and input grid"

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),k=nn,n_jobs=KNN_PROC)

        temp = self.sst()

        if nn==1:
            proj_temp = temp[m][ind]
            proj_temp[dist > md] = np.nan
            return proj_temp.reshape(tx.shape)
        else:
            inv_dist = 1/dist
            normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]
            proj_temp = temp[m][ind]
            proj_temp[dist > md] = np.nan
            return np.sum(proj_temp*normed_dist,axis=1).reshape(tx.shape)

class AMSR2_L1B(object):
    def __init__(self,path):
        def _CoRegParams(path):
            def _get_param_dict(ps):
                d = {}
                for i in range(len(ps)):
                    k,v = ps[i].split('-',1)
                    d[k] = float(v)
                return d

            h5 = tables.open_file(path,mode='r')
            root = h5.get_node('/')  
            a1 = _get_param_dict(root._v_attrs['CoRegistrationParameterA1'][0].split(','))
            a2 = _get_param_dict(root._v_attrs['CoRegistrationParameterA2'][0].split(','))
            h5.close()
            return a1, a2
        
        if not os.path.exists(path):
            raise RuntimeError, "AMSR2_L1B file [ %s ] does not exist" % (path)
        
        self.path = path
        self.a1, self.a2 = _CoRegParams(path)

    def latlon(self, band):
        def _CoReg(rlat,rlon,a1,a2):
            def _coRegScan(slat, slon, a1, a2):
                evn_lat, evn_lon = np.deg2rad(slat[::2]), np.deg2rad(slon[::2])
                odd_lat, odd_lon = np.deg2rad(slat[1::2]), np.deg2rad(slon[1::2])
                p1 = np.column_stack((np.cos(odd_lon)*np.cos(odd_lat),
                                        np.sin(odd_lon)*np.cos(odd_lat),
                                        np.sin(odd_lat)))
                p2 = np.column_stack((np.cos(evn_lon)*np.cos(evn_lat),
                                        np.sin(evn_lon)*np.cos(evn_lat),
                                        np.sin(evn_lat)))
                cp = np.cross(p1,p2)

                ex = p1.copy()
                ez = cp/np.linalg.norm(cp,axis=1)[:,np.newaxis]
                ey = np.cross(ez,ex)

                theta = np.arccos((p1*p2).sum(axis=1))[:,np.newaxis]
                pm = np.cos(a2*theta)*(np.cos(a1*theta)*ex+np.sin(a1*theta)*ey)+np.sin(a2*theta)*ez

                x,y,z = pm[:,0],pm[:,1],pm[:,2]
                lat = np.arcsin(z)
                lon = np.arctan2(y,x)

                return map(np.rad2deg,[lat,lon])

            rows = rlat.shape[0]
            regd = map(_coRegScan, [rlat[i,:] for i in range(rows)],
                                   [rlon[i,:] for i in range(rows)],
                                   [a1]*rows, [a2]*rows)
            return np.vstack([x[0] for x in regd]), np.vstack([x[1] for x in regd])

        h5 = tables.open_file(self.path,mode='r')
        lat89a = h5.get_node('/Latitude of Observation Point for 89A').read()
        lon89a = h5.get_node('/Longitude of Observation Point for 89A').read()
        h5.close()
        if band == '89.0':
            return lat89a, lon89a
        else:
            b_id = band.split('.')[0] + 'G'
            return _CoReg(lat89a, lon89a, self.a1[b_id], self.a2[b_id])

    def temperature(self, band, polarization):
        h5 = tables.open_file(self.path,mode='r')
        node = h5.get_node('/Brightness Temperature (%sGHz,%s)'%(band,polarization))
        temp = node.read() * node.attrs['SCALE FACTOR']
        h5.close()
        return temp

    def project(self, proj4_str, tx, ty, band, polarization, nn=2, md=20000):
        p = Proj(proj4_str)

        lat, lon = self.latlon(band)
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            raise ValueError, "no overlap between AMSR2 L1B data and input grid"

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),k=nn,n_jobs=KNN_PROC)
        temp = self.temperature(band, polarization)

        if nn==1:
            proj_temp = temp[m][ind]
            proj_temp[dist > md] = np.nan
            return proj_temp.reshape(tx.shape)
        else:
            inv_dist = 1/dist
            normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]
            proj_temp = temp[m][ind]
            proj_temp[dist > md] = np.nan
            return np.sum(proj_temp*normed_dist,axis=1).reshape(tx.shape)

class AMSR2_SIC(object):
    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "AMSR2 SIC file [ %s ] does not exist" % (path)
        self.path = path

    def latlon(self):
        h5 = tables.open_file(self.path,mode='r')
        lat = h5.get_node('/Latitude of Observation Point/').read()
        lon = h5.get_node('/Longitude of Observation Point/').read()
        h5.close()
        return lat, lon

    def sic(self):
        LAND_FLAG_VALUE = 32
        h5 = tables.open_file(self.path,mode='r')
        node = h5.get_node('/Geophysical Data/')
        data = np.squeeze(node.read()).astype('f8')
        data[data<-32766] = np.nan
        data *= node.attrs['SCALE FACTOR']
        land = np.squeeze(h5.get_node('/Pixel Data Quality/').read()) == LAND_FLAG_VALUE
        h5.close()
        data[land] = -1
        return data

    def project(self,proj4_str,tx,ty,nn=2,md=20000):
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 10:
            raise ValueError, "no overlap between AMSR2 SIC data and input grid"

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),k=nn,n_jobs=KNN_PROC)

        data = self.sic()

        if nn==1:
            proj_data = data[m][ind]
            proj_data[dist > md] = np.nan
            proj_data[proj_data < 0] = np.nan
            return proj_data.reshape(tx.shape)
        else:
            inv_dist = 1/dist
            normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]
            proj_data = data[m][ind]
            proj_data[dist > md] = np.nan
            proj_data = np.sum(proj_data*normed_dist,axis=1).reshape(tx.shape)
            proj_data[proj_data < 0] = np.nan
            return proj_data

class IMS_ASCII(object):
    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "IMS ASCII file [ %s ] does not exist" % (path)
        self.path = path
        self.lat_path = 'aux/ims1km_lat.bin'
        self.lon_path = 'aux/ims1km_lon.bin'
        self.res = 24576

    def latlon(self):
        lat = np.fromfile(self.lat_path)
        lat.shape = (self.res,self.res)

        lon = np.fromfile(self.lon_path)
        lon.shape = (self.res,self.res)

        return lat, lon

    def mask(self):

        fh = open(self.path,'r')

        # Skip Header
        for i in range(30):
            fh.readline()

        m = np.fromfile(fh, dtype='uint8')

        # Remove newlines
        m = m[m!=ord('\n')]

        return m.reshape(self.res,self.res) - ord('0')

    def project(self, proj4_str, tx, ty, md=2000):        
        #p = Proj(proj4_str)
        #lat, lon = self.latlon()
        #sx, sy = p(lon, lat)

        # DIRTY HACK TO SPEED THINGS UP
        # LOAD PRE-COMPUTED COORDINATES
        sx = np.fromfile('aux/ims1km_x.bin').reshape(self.res,self.res)
        sy = np.fromfile('aux/ims1km_y.bin').reshape(self.res,self.res)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between IMS ASCII data and input grid"
            return

        data = self.mask()

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        _, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),1)

        proj_band = data[m][ind]
        return proj_band.reshape(tx.shape)

class OSISAF_NC(object):
    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "OSISAF file [ %s ] does not exist" % (path)
        self.path = path

    def latlon(self):
        ds = netCDF4.Dataset(self.path)
        lat = np.array(ds.variables['lat'])
        lon = np.array(ds.variables['lon'])
        ds.close()
        return lat, lon

    def mask(self):
        """
        Flag Values:
        1: No Ice (less than 30 ice concentration)
        2: Open Ice (30-70 ice concentration)
        3: Closed Ice (more than 70 ice concentration)
        -1: FillValue
        """

        ds = netCDF4.Dataset(self.path)
        m = np.array(ds.variables['ice_edge']).squeeze()
        ds.close()
        return m

    def project(self, proj4_str, tx, ty):        
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between OSISAF_NC data and input grid"
            return

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        _, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),1)

        data = self.mask()

        proj_band = data[m][ind]
        return proj_band.reshape(tx.shape)

class NCEP_GRIB(object):

    def __init__(self, path, land_path="aux/seaice_sland127.grib"):
        if not os.path.exists(path):
            raise RuntimeError, "NCEP grib file [ %s ] does not exist" % (path)
        self.path = path
        self.land_path = land_path

    def latlon(self):
        g = pygrib.open(self.path)
        d = g.select()[0]
        lat, lon = d.latlons()
        g.close()
        return lat, lon

    def land(self):
        g = pygrib.open(self.land_path)
        d = g.select()[0]
        l = d.data()[0] > 0
        g.close()
        return l

    def sic(self):
        g = pygrib.open(self.path)
        d = g.select()[0]
        data = d.data()[0]
        data[(data>1)&(data<=1.28)] = 1
        data[data>1.28] = 0
        g.close()
        return data

    def project(self,proj4_str,tx,ty,nn=2,md=30000):
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between NCEP_GRIB data and input grid"
            return

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),nn)
        inv_dist = 1/dist
        normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]

        data = self.sic()
        proj_land = self.land()[m][ind[:,0]].reshape(tx.shape)

        #proj_data = data[m][ind[:,0]].reshape(tx.shape)
        proj_data = data[m][ind]
        proj_data[dist > md] = np.nan
        proj_data = np.sum(proj_data*normed_dist,axis=1).reshape(tx.shape)
        #proj_data = np.mean(proj_data,axis=1).reshape(tx.shape)
        proj_data[proj_land] = np.nan

        return proj_data

class CMC_NC(object):

    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "CMC file [ %s ] does not exist" % (path)
        self.path = path

    def latlon(self):
        return np.mgrid[-90:90.1:0.2,-180:180:0.2]


    def mask(self):
        """
        Flag Values:
        1: Water
        2: Land
        4: Lake
        8: Sea Ice
        """

        ds = netCDF4.Dataset(self.path)
        m = np.array(ds.variables['mask']).squeeze()
        ds.close()
        return m

    def project(self, proj4_str, tx, ty):        
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between CMC_NC data and input grid"
            return

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        _, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),1)

        data = self.mask()

        proj_band = data[m][ind]
        return proj_band.reshape(tx.shape)

class MASAM2_NC(object):
    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "MASAM2 file [ %s ] does not exist" % (path)
        self.path = path

    def latlon(self):
        ds = netCDF4.Dataset(self.path)
        lat = np.array(ds.variables['Latitude'][:])
        lon = np.array(ds.variables['Longitude'][:])
        ds.close()
        return lat, lon

    def mask(self,day):
        """
        Flag Values:
        0: No Ice (less than 70 ice concentration)
        70-100: Ice Concentration (70-100 ice concentration)
        104: Ocean out of AMSR2 bounds
        110: Intermediate or missing
        119: Land out of AMSR2 bounds
        120: Land
        """

        ds = netCDF4.Dataset(self.path)
        m = np.array(ds.variables['Sea_Ice_Concentration'][:])
        ds.close()
        return m[day-1]

    def project(self, proj4_str, tx, ty, data):        
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between MASAM2_NC data and input grid"
            return

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        _, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),1)

        proj_band = data[m][ind]
        return proj_band.reshape(tx.shape)    

class NIC_SHP(object):
    def __init__(self, path, land_path='aux/GSHHS_h_L1.shp'):
        if not os.path.exists(path):
            raise RuntimeError, "NIC shape file [ %s ] does not exist" % (path)
        self.path = path
        self.land_path = land_path

    def project(self,proj4_str,x0,y0,dx,dy,width,height):
        from osgeo import gdal, ogr, osr
        from tempfile import mkdtemp

        srs = osr.SpatialReference()
        srs.ImportFromProj4(proj4_str)

        rast_fn = mkdtemp()+'/nic_shp_tmp.tif'

        vect_ds = ogr.Open(self.path)
        vect = vect_ds.GetLayer()

        land_ds = ogr.Open(self.land_path)
        land = land_ds.GetLayer()

        rast = gdal.GetDriverByName('GTiff').Create(rast_fn,width,height,1,gdal.GDT_Byte)
        rast.SetProjection(srs.ExportToWkt())
        rast.SetGeoTransform((x0,dx,0,y0,0,dy))
        rast_band = rast.GetRasterBand(1)
        rast_band.SetNoDataValue(0)

        gdal.RasterizeLayer(rast,[1],vect,burn_values=[1])
        gdal.RasterizeLayer(rast,[1],land,burn_values=[2])

        mask = rast_band.ReadAsArray().astype('uint8')

        vect_ds = None
        land_ds = None
        rast = None

        os.remove(rast_fn)
        os.rmdir(rast_fn[:-16])

        return mask

class IMS_GRIB(object):

    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError, "IMS grib file [ %s ] does not exist" % (path)
        self.path = path

    def latlon(self):
        g = pygrib.open(self.path)
        d = g.select()[0]
        lat, lon = d.latlons()
        g.close()
        return lat, lon

    def land(self):
        g = pygrib.open(self.path)
        d = g.select()[0]['values']
        l = d.mask
        g.close()
        return l

    def sic(self):
        g = pygrib.open(self.path)
        d = g.select()[0]['values']
        data = d.data
        data[data==d.fill_value] = np.nan
        g.close()
        return data

    def project(self,proj4_str,tx,ty,nn=2,md=8000):
        p = Proj(proj4_str)

        lat, lon = self.latlon()
        sx, sy = p(lon, lat)
        
        data = self.sic()

        m = (sx>=tx.min())&(sx<=tx.max())&(sy>=ty.min())&(sy<=ty.max())
        
        if np.sum(m) < 100:
            print "no overlap between data and input grid"
            return

        k = cKDTree(np.column_stack([sx[m],sy[m]]))
        dist, ind = k.query(np.column_stack([tx.ravel(),ty.ravel()]),nn)
        inv_dist = 1/dist
        normed_dist = inv_dist / np.sum(inv_dist,axis=1)[:,np.newaxis]

        proj_land = self.land()[m][ind[:,0]].reshape(tx.shape)

        #proj_data = data[m][ind[:,0]].reshape(tx.shape)
        proj_data = data[m][ind]
        proj_data[dist > md] = np.nan
        proj_data = np.nansum(proj_data*normed_dist,axis=1).reshape(tx.shape)
        #proj_data = np.mean(proj_data,axis=1).reshape(tx.shape)
        proj_data[proj_land] = np.nan

        return proj_data
