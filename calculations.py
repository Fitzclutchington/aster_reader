import numpy as np

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

    p = np.polyfit(bins,np.arange(1,l),1)
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