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
  radius = 6371
  lat = lat = hdf.select('Latitude').get()
  lon = hdf.select('Longitude').get()
  
  h_step = 492
  v_step = 420
  base_coords = grid.degreeToSphere(lat,lon,radius)
  corners = np.array([[base_coords[0,0,0],base_coords[0,1,0]],[base_coords[1,0,0],base_coords[1,1,0]]])
  surface = grid.bilinearInterp(corners, h_step, v_step)
  full_coords = grid.geoInterp(base_coords, h_step, v_step)
  geo_coords = grid.sphereToDegree(full_coords,radius)
  
  ds = gdal.Open('HDF4_EOS:EOS_SWATH:"AST_L1B_00307182015230811_20150720125749_13236.hdf":VNIR_Swath:ImageData1')
  tmp_ds=gdal.AutoCreateWarpedVRT(ds)
  projection=tmp_ds.GetProjection()
  osrref = osr.SpatialReference()
  osrref.ImportFromWkt(projection)
  projstr = osrref.ExportToProj4()
  utm = Proj(projstr)
  
  tx,ty = utm(geo_coords[:,:,1],geo_coords[:,:,0])

  """
  plt.figure()
  plt.imshow(full_coords[:,:,0])
  plt.colorbar()
  
  plt.figure()
  plt.imshow(full_coords[:,:,1])
  plt.colorbar()
  
  plt.figure()
  plt.imshow(full_coords[:,:,2])
  plt.colorbar() 
  """

  b1 = hdf.select('ImageData1')
  db1 = b1.get()
  reflectance_b1 = RadToRefl ( DnToRad ( db1, 1, getGain(hdf,1) ), 1, earth_sun_dist, sza)
  print reflectance_b1.shape
  rfb1 = reflectance_b1 / np.cos(sza)
  
  b2 = hdf.select('ImageData2')
  db2 = b2.get()
  reflectance_b2 = RadToRefl ( DnToRad ( db2, 2, getGain(hdf,2)), 2, earth_sun_dist, sza)
  rfb2 = reflectance_b2 / np.cos(sza)

  b3N = hdf.select('ImageData3N')
  db3N = b3N.get()
  reflectance_b3 = RadToRefl ( DnToRad ( db3N, 3, getGain(hdf,'3N')), 3, earth_sun_dist, sza)
  rfb3 = reflectance_b3 / np.cos(sza)
  
  #np.dstack((rfb1,rfb2,rfb3))
  """
  plt.figure()
  plt.imshow ( reflectance_b1 ) #vmin=-0.2, vmax=0.8, interpolation='nearest')
  plt.colorbar()
  plt.show()
  """
