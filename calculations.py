import numpy as np
from pykdtree.kdtree import KDTree

def DnToRad ( data, band, gain ):
  """formula and table come from pg 25,26 in ASTER man"""

  gain_settings={'HGH':0, 'NOR':1, 'low1':2, 'low2':3}
  conversion_factors=np.array([[0.676, 1.688, 2.25, 0.0],\
              [0.708, 1.4105, 1.89, 0.0],\
              [0.423, 0.862, 1.15, 0.0],\
              [0.1087, 0.2174, 0.2900, 0.2900],\
              [0.0348, 0.0696, 0.0925, 0.4090],\
              [0.0313, 0.0625, 0.0830, 0.3900],\
              [0.0299, 0.0597, 0.0795, 0.3320],\
              [0.0209, 0.0417, 0.0556, 0.2450],\
              [0.0159, 0.0318, 0.0424, 0.2650]])
  radiance = (data-1.)*conversion_factors[band-1][gain_settings[gain]]
  return radiance

def RadToRefl ( radiance, band, dist, sza, solar_constant_scheme="ThomeEtAl2" ):
  schemes={'Smith':0, 'ThomeEtAl1':1, 'ThomeEtAl2':2}
  solar_constant=np.array([[1845.99, 1847, 1848],    \
                  [1555.74, 1553, 1549],    \
                  [1119.47, 1118, 1114],    \
                  [231.25, 232.5,  225.4 ], \
                  [79.81,  80.32,  86.63 ], \
                  [74.99,  74.92,  81.85 ], \
                  [68.66,  69.20,  74.85 ], \
                  [59.74,  59.82,  66.49 ], \
                  [56.92,  57.32,  59.85]]) #B9
  # don't know where this equation comes from
  #d = 1.0 - 0.01672*np.cos(np.radians(0.9856*(day-4)))
  toaRefl = (np.pi*radiance*dist*dist)/(solar_constant[band-1][schemes[solar_constant_scheme]]*\
        np.cos ( np.radians (sza) ))
  return toaRefl

def hist_match(source, template):
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

def hist_match2(Z,h,bins):
    l = len(bins)
    g = np.nan*np.ones((1,l))
    cdf = np.cumsum(h)

    hQ,_ = np.histogram(Z,bins=bins)
    cdfQ = np.cumsum(hQ)

    c=0; kQ=1
    while (c<np.sum(hQ)) and (kQ<l-1):
        kQ = np.min(np.where(cdfQ>c))
        k = np.max(np.where(cdf<cdfQ[kQ]))
        g[kQ]=bins[k+1]
        kQ+=1
        c=cdfQ[kQ]
    
    g[~np.isfinite(g)] = np.interp(bins[~np.isfinite(g)],bins[np.isfinite(g)],g[isfinite(g)])

    p = np.polyfit(bins,np.arange(1,l+1),1)
    idx = np.round(np.polyval(p,Z))
    return g[idx]

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def pca(Q, ncomp):
    """Principal component analysis.

    Returns `ncomp` principal compenents of `Q`.

    """
    Q = Q - np.mean(Q, axis=0)
    w, v = np.linalg.eig(np.cov(Q.T))
    eigorder = w.argsort()
    return np.dot(Q, v[:,eigorder[-ncomp:]])

def hist_match3(source,target,eps,nbins=100):
    D1 = target.ravel()
    #D1_mean = np.mean(D1)
    #D1_std = np.std(D1)
    #out1_pos = (D1_mean+eps)*D1_std
    #out1_neg = (D1_mean-eps)*D1_std
    #m1 = D1 > out1_pos
    #m2 = D1 < out1_neg
    #out1_mask = np.logical_or(m1,m2)
    m = D1.min()
    M = D1.max()
    H = np.zeros((2, nbins))


    D = source.copy()
    D_shape = D.shape
    D = D.ravel()
    #D_mean = np.mean(D)
    #D_std = np.std(D)
    #out2_pos = (D_mean+eps)*D_std
    #out2_neg = (D_mean-eps)*D_std
    #m1 = D > out2_pos
    #m2 = D < out2_neg
    #out2_mask = np.logical_or(m1,m2)

    x = np.linspace(min(m, np.min(D)), max(M, np.max(D)), nbins+1)
    WH, _ = np.histogram(D1, bins=x, normed=True)
    H[0,:] = np.cumsum(WH/float(len(D1)))
    WH, _ = np.histogram(D, bins=x, normed=True)
    H[1,:] = np.cumsum(WH/float(len(D)))
    y = np.zeros(nbins)
    for i in xrange(nbins):
        indL = np.where(H[0,:] <= H[1,i])[0]
        if len(indL) == 0 or len(indL) == nbins:
            y[i] = x[i]
        else:
            pos = indL.max()
            xL = x[pos]
            fL = H[0, pos]
            xR = x[pos+1]
            fR = H[0, pos+1]
            y[i] = xL + (H[1,i]-fL)*(xR-xL)/float(fR-fL)

    B = np.interp(D, x[:-1], y)
    
    return B.reshape(D_shape)

def desaturate_aster(aster,aster_nonsat,modis):

  Q = np.column_stack((aster,aster_nonsat,modis))
  ms = np.mean(Q,axis=0)
  d,v = np.linalg.eig(np.cov(Q.T))
  eigorder = d.argsort()
  Pr = np.dot(np.dot((Q-ms),v[:,eigorder[-1]])[:,np.newaxis],v[:,eigorder[-1]].reshape((1,3))) + ms
  #Pr = np.dot(np.dot((Q-ms),v[:,eigorder[-1]]).reshape((Q.shape[0],1)),v[:,eigorder[-1]].reshape((1,2))) + ms
  return Pr[:,0]


def getBlueAster(aster_match,modis_bands,edge_mask,shape):
  aster = np.column_stack(aster_match)
  modis = np.column_stack(modis_bands[:-1])
  ms = np.mean(np.vstack((aster,modis)),axis=0)
  Mh = aster - ms
  M = modis - ms
  d,v = np.linalg.eig(np.cov(Mh.T))
  eigorder = d.argsort()
  aster_eig = v[:,eigorder[-1]]
  
  d,v = np.linalg.eig(np.cov(M.T))
  eigorder = d.argsort()
  modis_eig = v[:,eigorder[-3:]]
  modis_eig[:,-1] = aster_eig
  CT = np.dot(M,modis_eig)
  
  Mhat = np.dot(CT,modis_eig.T) +ms

  """
  A = np.column_stack((np.ones(Mhat.shape[0]),Mhat))
  b = modis_bands[3]
  x = np.linalg.lstsq(A, b)[0]
  
  aster_blue = np.dot(np.column_stack((np.ones(aster.shape[0]),aster)),x)
  """
  print "starting kd tree"
  k = KDTree(np.column_stack(modis[:-1]))
  print "KDTree built"
  dist, ind = k.query(np.column_stack(aster))
  print "cKDTree query completed"
  #nearest neighbor
  nnvals = modis[3][ind]
   
  aster_blue = np.zeros(shape)
  aster_blue[~edge_mask] = nnvals
  aster_blue[edge_mask] = np.nan

  aster_test = np.zeros((shape[0],shape[1],3))
  aster_test[~edge_mask,:] = np.column_stack((aster_match[1],aster_match[0],aster_blue))
  aster_test[aster_test<0] = 0
  aster_test[aster_test>1] = 1
  aster_test[edge_mask,:] = np.nan
  return aster_test

  
  

  
