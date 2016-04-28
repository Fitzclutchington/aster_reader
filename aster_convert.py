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
from scipy.spatial import cKDTree
from scipy.io import savemat
import calculations as calc
import utils

if __name__=="__main__":
   
  filename = sys.argv[1] 

  hdf = utils.openHDF(filename)
  julian_date = utils.filenameToJulian(filename)
  
  earth_sun_dist = utils.sunDistance(julian_date)
  sza = utils.getSZA(hdf)
  
  # currently using volumetric mean radius from:
  # http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
  
  lat = hdf.select('Latitude').get()
  lon = hdf.select('Longitude').get()
  radius = utils.getRadius(np.deg2rad(lat))
  
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
  reflectance_b1 = calc.RadToRefl ( calc.DnToRad ( db1, 1, utils.getGain(hdf,1) ), 1, earth_sun_dist, sza)
  rfb1 = reflectance_b1 / np.cos(sza)
  
  b2 = hdf.select('ImageData2')
  db2 = b2.get().astype('f8')
  db2[edge_mask] = np.nan
  reflectance_b2 = calc.RadToRefl ( calc.DnToRad ( db2, 2, utils.getGain(hdf,2)), 2, earth_sun_dist, sza)
  rfb2 = reflectance_b2 / np.cos(sza)

  b3N = hdf.select('ImageData3N')
  db3N = b3N.get().astype('f8') 
  db3N[edge_mask] = np.nan
  reflectance_b3 = calc.RadToRefl ( calc.DnToRad ( db3N, 3, utils.getGain(hdf,'3N')), 3, earth_sun_dist, sza)
  rfb3 = reflectance_b3 / np.cos(sza)
  
  rfb1_match = calc.hist_match(reflectance_b1,pbands[3])
  rfb1_match[edge_mask] = np.nan
  rfb2_match = calc.hist_match(reflectance_b2,pbands[0])
  rfb2_match[edge_mask] = np.nan
  rfb3_match = calc.hist_match(reflectance_b3,pbands[1])
  rfb3_match[edge_mask] = np.nan  
  
  k = cKDTree(np.column_stack([pbands[3].ravel(),pbands[0].ravel(),pbands[1].ravel()]))
  dist, ind = k.query(np.column_stack([rfb1_match[~edge_mask],rfb2_match[~edge_mask],rfb3_match[~edge_mask]]),n_jobs=20, eps=0.1)
  
  #nearest neighbor
  nnvals = pbands[2][ind]
  
  aster_blue = np.zeros(rfb1_match.shape)
  aster_blue[~edge_mask] = nnvals
  aster_blue[edge_mask] = np.nan

  plt.figure()
  plt.imshow(np.dstack([rfb2_match,rfb1_match,aster_blue]))
  plt.colorbar()

  
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
  
  x1, y1 = calc.ecdf(reflectance_b1.ravel())
  x2, y2 = calc.ecdf(pbands[3].ravel())
  x3, y3 = calc.ecdf(rfb1_match.ravel())

  axarr[2,0].plot(x1,y1,'-r',label="aster")
  axarr[2,0].plot(x2,y2,'-k',label="modis")
  axarr[2,0].plot(x3,y3,'--b',lw=3,label="match")
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
  
  x1, y1 = calc.ecdf(reflectance_b2.ravel())
  x2, y2 = calc.ecdf(pbands[0].ravel())
  x3, y3 = calc.ecdf(rfb2_match.ravel())

  axarr[2,1].plot(x1,y1,'-r',label="aster")
  axarr[2,1].plot(x2,y2,'-k',label="modis")
  axarr[2,1].plot(x3,y3,'--b',lw=3,label="match")
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
  
  x1, y1 = calc.ecdf(reflectance_b3.ravel())
  x2, y2 = calc.ecdf(pbands[1].ravel())
  x3, y3 = calc.ecdf(rfb3_match.ravel())

  axarr[2,2].plot(x1,y1,'-r',label="aster")
  axarr[2,2].plot(x2,y2,'-k',label="modis")
  axarr[2,2].plot(x3,y3,'--b',lw=3,label="match")
  axarr[2,2].set_title('Aster 3 Modis 2 Matched Histogram')
  axarr[2,2].set_xlabel('Pixel value')
  axarr[2,2].set_ylabel('Cumulative %')
  axarr[2,2].legend(loc=5)

  fig.savefig('modis_aster_projection.png')
  plt.close()