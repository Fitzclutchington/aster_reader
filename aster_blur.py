# -*- coding: utf-8 -*-
from osgeo import gdal, osr
from pyproj import Proj
import numpy as np
import matplotlib.pyplot as plt
from pyhdf import SD
import sys
import gridding as grid
from products import MODIS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import savemat
import calculations as calc
import utils
import json
from scipy import misc

if __name__=="__main__":
   
  filename = sys.argv[1] 
  with open(filename) as f:
    config = json.load(f)

  aster_file = str(config['aster'])
  mhkm = str(config['mhkm'])
  m1km = str(config['m1km'])
  m03 =  str(config['m03'])
  file_end = str(config['file_end'])

  hdf = utils.openHDF(aster_file)
  julian_date = utils.filenameToJulian(aster_file)
  
  earth_sun_dist = utils.sunDistance(julian_date)
  sza = utils.getSZA(hdf)
  
  # currently using volumetric mean radius from:
  # http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
  b1 = hdf.select('ImageData1')
  db1 = b1.get().astype('f8')
  edge_mask = db1==0
  db1[edge_mask] = np.nan  
  reflectance_b1 = calc.RadToRefl ( calc.DnToRad ( db1, 1, utils.getGain(hdf,1) ), 1, earth_sun_dist, sza)

  
  b2 = hdf.select('ImageData2')
  db2 = b2.get().astype('f8')
  db2[edge_mask] = np.nan
  reflectance_b2 = calc.RadToRefl ( calc.DnToRad ( db2, 2, utils.getGain(hdf,2)), 2, earth_sun_dist, sza)

  b3N = hdf.select('ImageData3N')
  db3N = b3N.get().astype('f8') 
  db3N[edge_mask] = np.nan
  reflectance_b3 = calc.RadToRefl ( calc.DnToRad ( db3N, 3, utils.getGain(hdf,'3N')), 3, earth_sun_dist, sza)
  

  lat = hdf.select('Latitude').get()
  lon = hdf.select('Longitude').get()
  radius = utils.getRadius(np.deg2rad(lat))
  
  h_step = int(reflectance_b1.shape[1]/10)
  v_step = int(reflectance_b1.shape[0]/10)
  base_coords = grid.degreeToSphere(lat,lon,radius)
  corners = np.array([[base_coords[0,0,0],base_coords[0,1,0]],[base_coords[1,0,0],base_coords[1,1,0]]])
  #surface = grid.bilinearInterp(corners, h_step, v_step)
  full_coords = grid.geoInterp(base_coords, h_step, v_step)
  geo_coords = grid.sphereToDegree(full_coords)
  
  geo_coords[:,:,0] = grid.toGeocentric(geo_coords[:,:,0])
  
  ds = gdal.Open('HDF4_EOS:EOS_SWATH:"{}":VNIR_Swath:ImageData1'.format(aster_file))
  tmp_ds=gdal.AutoCreateWarpedVRT(ds)
  projection=tmp_ds.GetProjection()
  osrref = osr.SpatialReference()
  osrref.ImportFromWkt(projection)
  projstr = osrref.ExportToProj4()
  utm = Proj(projstr)

  tx,ty = utm(geo_coords[:,:,1],geo_coords[:,:,0])
  
  mobj = MODIS(mhkm,m1km,m03)
  
  bands = mobj.reflectance([1,2,3,4])

  pbands = mobj.project(projstr,tx,ty,bands,nn=1)

  print "tx (1,0) - (0,0) = " + str(tx[1,0]-tx[0,0])
  print "tx (0,1) - (0,0) = " + str(tx[0,1]-tx[0,0])

  print "ty (1,0) - (0,0) = " + str(ty[1,0]-ty[0,0])
  print "ty (0,1) - (0,0) = " + str(ty[0,1]-ty[0,0])