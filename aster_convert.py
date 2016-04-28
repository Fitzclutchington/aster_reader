# -*- coding: utf-8 -*-
import os
from osgeo import gdal, osr
from pyproj import Proj
import numpy as np
import pandas as pd
from pyhdf import SD
import matplotlib.pyplot as plt
import sys
from calendar import monthrange
import odl_parser
import gridding as grid
from products import MODIS
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

def filenameToJulian(filename):
    file_string = filename.split('_')
    if file_string[0] != 'AST':
      print 'Non-Aster File'
      sys.exit(1)
    date_string = file_string[2]
    month = int(date_string[3:5])
    day = int(date_string[5:7])
    year = int(date_string[7:11])
    return dateToJulian(month,day,year)

def dateToJulian(month,day,year):
  julianDay = 0
  for i in range(1,month):
    julianDay+= monthrange(year,month)[1]
  julianDay += day
  return julianDay

def sunDistance(julian_date):
  d_file = pd.read_csv('earth_sun_distances.csv')
  distance_dict = d_file.set_index('day').to_dict()['distance']
  return float(distance_dict[julian_date])

def getSZA(hdf):
  meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])['ASTERGENERICMETADATA']
  scene_info = meta['SCENEINFORMATION']
  sza = scene_info['SOLARDIRECTION'][1]
  return 90 - float(sza)

def getGain(hdf,band):
  meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])['ASTERGENERICMETADATA']
  gain_info = meta['GAININFORMATION']['GAIN']
  gain_info = [g.split(",") for g in gain_info]
  gain_info = [g[1].strip().strip(")").strip('"') for g in gain_info]
  print gain_info
  if band < 3:
    gain = gain_info[band-1]
  elif band == '3N':
    gain = gain_info[2]
  elif band == '3B':
    gain = gain_info[3]
  elif band > 3 and band < 10:
    gain = gain_info[band]
  else:
    print "band not recognized"
    sys.exit(1)
  return gain

def getRadius(lat):

  a = 6384.4*1000
  b = 6352.8*1000
  return np.sqrt((np.square(a*a*np.cos(lat)) + np.square(b*b*np.sin(lat)))/ \
                 (np.square(a*np.cos(lat)) + np.square(b*np.sin(lat))) )

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

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

if __name__=="__main__":
   
  filename = sys.argv[1] 
  fil, file_extension = os.path.splitext(filename)

  if file_extension != '.hdf':
    print "Not an HDF file"
    sys.exit(1)

  hdf = SD.SD(filename)
  julian_date = filenameToJulian(filename)
  
  earth_sun_dist = sunDistance(julian_date)
  sza = getSZA(hdf)
  
  # currently using volumetric mean radius from:
  # http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
  
  lat = hdf.select('Latitude').get()
  lon = hdf.select('Longitude').get()
  radius = getRadius(np.deg2rad(lat))
  
  h_step = 492
  v_step = 420
  base_coords = grid.degreeToSphere(lat,lon,radius)
  corners = np.array([[base_coords[0,0,0],base_coords[0,1,0]],[base_coords[1,0,0],base_coords[1,1,0]]])
  #surface = grid.bilinearInterp(corners, h_step, v_step)
  full_coords = grid.geoInterp(base_coords, h_step, v_step)
  geo_coords = grid.sphereToDegree(full_coords)
  
  geo_coords[:,:,0] = grid.toGeocentric(geo_coords[:,:,0])
  
  ds = gdal.Open('HDF4_EOS:EOS_SWATH:"{}":VNIR_Swath:ImageData1'.format(filename))
  tmp_ds=gdal.AutoCreateWarpedVRT(ds)
  projection=tmp_ds.GetProjection()
  osrref = osr.SpatialReference()
  osrref.ImportFromWkt(projection)
  projstr = osrref.ExportToProj4()
  utm = Proj(projstr)
  
  tx,ty = utm(geo_coords[:,:,1],geo_coords[:,:,0])
  
  mhkm = 'modis/MOD02HKM.A2015158.2310.006.2015159075416.hdf'
  m1km = 'modis/MOD021KM.A2015158.2310.006.2015159075416.hdf'
  m03 = 'modis/MOD03.A2015158.2310.006.2015159052213.hdf'
  mobj = MODIS(mhkm,m1km,m03)

  bands = mobj.reflectance([1,2,3,4])

  pbands = mobj.project(projstr,tx,ty,bands,nn=1)
    

  b1 = hdf.select('ImageData1')
  db1 = b1.get().astype('f8')
  edge_mask = db1==0
  db1[edge_mask] = np.nan  
  reflectance_b1 = RadToRefl ( DnToRad ( db1, 1, getGain(hdf,1) ), 1, earth_sun_dist, sza)
  rfb1 = reflectance_b1 / np.cos(sza)
  
  b2 = hdf.select('ImageData2')
  db2 = b2.get().astype('f8')
  db2[edge_mask] = np.nan
  reflectance_b2 = RadToRefl ( DnToRad ( db2, 2, getGain(hdf,2)), 2, earth_sun_dist, sza)
  rfb2 = reflectance_b2 / np.cos(sza)

  b3N = hdf.select('ImageData3N')
  db3N = b3N.get().astype('f8') 
  db3N[edge_mask] = np.nan
  reflectance_b3 = RadToRefl ( DnToRad ( db3N, 3, getGain(hdf,'3N')), 3, earth_sun_dist, sza)
  rfb3 = reflectance_b3 / np.cos(sza)
  
  rfb1_match = hist_match(reflectance_b1,pbands[3])
  rfb1_match[edge_mask] = np.nan
  rfb2_match = hist_match(reflectance_b2,pbands[0])
  rfb2_match[edge_mask] = np.nan
  rfb3_match = hist_match(reflectance_b3,pbands[1])
  rfb3_match[edge_mask] = np.nan  

  fig, axarr = plt.subplots(3,3, figsize=(30,20))
  img1 = axarr[0,0].imshow(pbands[3], vmin=0,vmax=1)
  axarr[0,0].set_title('Modis band 4')
  div1 = make_axes_locatable(axarr[0,0])
  cax1 = div1.append_axes("right", size="15%", pad=0.05)
  cbar1 = plt.colorbar(img1, cax=cax1)
  axarr[0,0].xaxis.set_visible(False)
  axarr[0,0].yaxis.set_visible(False)

  

  img2 = axarr[1,0].imshow(rfb1_match, vmin=0,vmax=1)
  axarr[1,0].set_title('Aster band 1')
  div2 = make_axes_locatable(axarr[1,0])
  cax2 = div2.append_axes("right", size="15%", pad=0.05)
  cbar2 = plt.colorbar(img2, cax=cax2)
  axarr[1,0].xaxis.set_visible(False)
  axarr[1,0].yaxis.set_visible(False)
  
  x1, y1 = ecdf(reflectance_b1.ravel())
  x2, y2 = ecdf(pbands[3].ravel())
  x3, y3 = ecdf(rfb1_match.ravel())

  axarr[2,0].plot(x1,y1,'-r',label="aster")
  axarr[2,0].plot(x2,y2,'-k',label="modis")
  axarr[2,0].plot(x3,y3,'--b',label="match")
  axarr[2,0].set_title('Aster 1 Modis 4 Matched Histogram')
  axarr[2,0].set_xlabel('Pixel value')
  axarr[2,0].set_ylabel('Cumulative %')
  axarr[2,0].legend(loc=5)

  img3 = axarr[0,1].imshow(pbands[0], vmin=0,vmax=1)
  axarr[0,1].set_title('Modis band 1')
  div3 = make_axes_locatable(axarr[0,1])
  cax3 = div3.append_axes("right", size="15%", pad=0.05)
  cbar3 = plt.colorbar(img3, cax=cax3)
  axarr[0,1].xaxis.set_visible(False)
  axarr[0,1].yaxis.set_visible(False)

  img4 = axarr[1,1].imshow(rfb2_match, vmin=0,vmax=1)
  axarr[1,1].set_title('Aster band 2')
  div4 = make_axes_locatable(axarr[1,1])
  cax4 = div4.append_axes("right", size="15%", pad=0.05)
  cbar4 = plt.colorbar(img4, cax=cax4)
  axarr[1,1].xaxis.set_visible(False)
  axarr[1,1].yaxis.set_visible(False)
  
  x1, y1 = ecdf(reflectance_b2.ravel())
  x2, y2 = ecdf(pbands[0].ravel())
  x3, y3 = ecdf(rfb2_match.ravel())

  axarr[2,1].plot(x1,y1,'-r',label="aster")
  axarr[2,1].plot(x2,y2,'-k',label="modis")
  axarr[2,1].plot(x3,y3,'--b',label="match")
  axarr[2,1].set_title('Aster 2 Modis 1 Matched Histogram')
  axarr[2,1].set_xlabel('Pixel value')
  axarr[2,1].set_ylabel('Cumulative %')
  axarr[2,1].legend(loc=5)

  img5 = axarr[0,2].imshow(pbands[1], vmin=0,vmax=1)
  axarr[0,2].set_title('Modis band 2')
  div5 = make_axes_locatable(axarr[0,2])
  cax5 = div5.append_axes("right", size="15%", pad=0.05)
  cbar5 = plt.colorbar(img5, cax=cax5)
  axarr[0,2].xaxis.set_visible(False)
  axarr[0,2].yaxis.set_visible(False)


  img6 = axarr[1,2].imshow(rfb3_match, vmin=0,vmax=1)
  axarr[1,2].set_title('Aster band 3N')
  div6 = make_axes_locatable(axarr[1,2])
  cax6 = div6.append_axes("right", size="15%", pad=0.05)
  cbar6 = plt.colorbar(img6, cax=cax6)
  axarr[1,2].xaxis.set_visible(False)
  axarr[1,2].yaxis.set_visible(False)
  
  x1, y1 = ecdf(reflectance_b3.ravel())
  x2, y2 = ecdf(pbands[1].ravel())
  x3, y3 = ecdf(rfb3_match.ravel())

  axarr[2,2].plot(x1,y1,'-r',label="aster")
  axarr[2,2].plot(x2,y2,'-k',label="modis")
  axarr[2,2].plot(x3,y3,'--b',label="match")
  axarr[2,2].set_title('Aster 3 Modis 2 Matched Histogram')
  axarr[2,2].set_xlabel('Pixel value')
  axarr[2,2].set_ylabel('Cumulative %')
  axarr[2,2].legend(loc=5)

  fig.savefig('modis_aster_projection.png')
  plt.close()
  