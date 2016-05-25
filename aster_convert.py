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
  color_img = np.dstack((pbands[0],pbands[3],pbands[2]))
  
  #plt.figure()
  #plt.imshow(color_img)
  #plt.colorbar()
  #plt.show()  
"""
  rfb1_proj = np.zeros(reflectance_b1.shape)
  rfb1_proj[~edge_mask] =  calc.desaturate_aster(reflectance_b1[~edge_mask],pbands[3][~edge_mask])
  rfb1_proj[edge_mask] = np.nan

  rfb2_proj = np.zeros(reflectance_b1.shape)
  rfb2_proj[~edge_mask] =  calc.desaturate_aster(reflectance_b2[~edge_mask],pbands[0][~edge_mask])
  rfb2_proj[edge_mask] = np.nan
"""

  rfb1_proj = np.zeros(reflectance_b1.shape)
  rfb1_proj[~edge_mask] =  calc.desaturate_aster(reflectance_b1[~edge_mask],reflectance_b3[~edge_mask])
  rfb1_proj[edge_mask] = np.nan

  rfb2_proj = np.zeros(reflectance_b1.shape)
  rfb2_proj[~edge_mask] =  calc.desaturate_aster(reflectance_b2[~edge_mask],reflectance_b3[~edge_mask])
  rfb2_proj[edge_mask] = np.nan
  
  
  #used in hist matchingto get rid of outliers
  eps = 3.5
  rfb1_match = calc.hist_match3(rfb1_proj[~edge_mask],pbands[3][~edge_mask],eps,nbins=200)
  #rfb1_match[edge_mask] = np.nan
  rfb2_match = calc.hist_match3(rfb2_proj[~edge_mask],pbands[0][~edge_mask],eps,nbins=200)
  #rfb2_match[edge_mask] = np.nan
  rfb3_match = calc.hist_match3(reflectance_b3[~edge_mask],pbands[1][~edge_mask],eps,nbins=200)
  #rfb3_match[edge_mask] = np.nan
  
  rfb1_match_full = np.zeros(edge_mask.shape)
  rfb1_match_full[~edge_mask]=rfb1_match
  rfb1_match_full[edge_mask]=np.nan
  
  rfb2_match_full = np.zeros(reflectance_b1.shape)
  rfb2_match_full[~edge_mask]=rfb2_match
  rfb2_match_full[edge_mask]=np.nan

  rfb3_match_full = np.zeros(reflectance_b1.shape)
  rfb3_match_full[~edge_mask]=rfb3_match
  rfb3_match_full[edge_mask]=np.nan
  
  
  fig, axarr = plt.subplots(3,3, figsize=(30,20))
  img1 = axarr[0,0].imshow(pbands[3], vmin=0,vmax=1)
  axarr[0,0].set_title('Modis band 4')
  div1 = make_axes_locatable(axarr[0,0])
  cax1 = div1.append_axes("right", size="10%", pad=0.05)
  cbar1 = plt.colorbar(img1, cax=cax1)
  axarr[0,0].xaxis.set_visible(False)
  axarr[0,0].yaxis.set_visible(False)

  

  img2 = axarr[1,0].imshow(rfb1_match_full, vmin=0,vmax=1)
  axarr[1,0].set_title('Aster band 1')
  div2 = make_axes_locatable(axarr[1,0])
  cax2 = div2.append_axes("right", size="10%", pad=0.05)
  cbar2 = plt.colorbar(img2, cax=cax2)
  axarr[1,0].xaxis.set_visible(False)
  axarr[1,0].yaxis.set_visible(False)
  
  x1, y1 = calc.ecdf(rfb1_proj.ravel())
  x2, y2 = calc.ecdf(pbands[3].ravel())
  x3, y3 = calc.ecdf(rfb1_match.ravel())
  x4, y4 = calc.ecdf(reflectance_b1.ravel())

  axarr[2,0].plot(x1,y1,'-r',label="aster")
  axarr[2,0].plot(x2,y2,'-k',label="modis")
  axarr[2,0].plot(x3,y3,'--b',lw=3,label="match")
  axarr[2,0].plot(x4,y4,'g', label="sat aster")
  axarr[2,0].set_title('Aster 1 Modis 4 Matched Histogram')
  axarr[2,0].set_xlabel('Reflectance')
  axarr[2,0].set_ylabel('Cumulative %')
  axarr[2,0].legend(loc=5)

  img3 = axarr[0,1].imshow(pbands[0], vmin=0,vmax=1)
  axarr[0,1].set_title('Modis band 1')
  div3 = make_axes_locatable(axarr[0,1])
  cax3 = div3.append_axes("right", size="10%", pad=0.05)
  cbar3 = plt.colorbar(img3, cax=cax3)
  axarr[0,1].xaxis.set_visible(False)
  axarr[0,1].yaxis.set_visible(False)

  img4 = axarr[1,1].imshow(rfb2_match_full, vmin=0,vmax=1)
  axarr[1,1].set_title('Aster band 2')
  div4 = make_axes_locatable(axarr[1,1])
  cax4 = div4.append_axes("right", size="10%", pad=0.05)
  cbar4 = plt.colorbar(img4, cax=cax4)
  axarr[1,1].xaxis.set_visible(False)
  axarr[1,1].yaxis.set_visible(False)
  
  x1, y1 = calc.ecdf(rfb2_proj.ravel())
  x2, y2 = calc.ecdf(pbands[0].ravel())
  x3, y3 = calc.ecdf(rfb2_match.ravel())
  x4, y4 = calc.ecdf(reflectance_b2.ravel())

  axarr[2,1].plot(x1,y1,'-r',label="aster")
  axarr[2,1].plot(x2,y2,'-k',label="modis")
  axarr[2,1].plot(x3,y3,'--b',lw=3,label="match")
  axarr[2,1].plot(x4,y4,'g', label="sat aster")
  axarr[2,1].set_title('Aster 2 Modis 1 Matched Histogram')
  axarr[2,1].set_xlabel('Reflectance')
  axarr[2,1].set_ylabel('Cumulative %')
  axarr[2,1].legend(loc=5)

  img5 = axarr[0,2].imshow(pbands[1], vmin=0,vmax=1)
  axarr[0,2].set_title('Modis band 2')
  div5 = make_axes_locatable(axarr[0,2])
  cax5 = div5.append_axes("right", size="10%", pad=0.05)
  cbar5 = plt.colorbar(img5, cax=cax5)
  axarr[0,2].xaxis.set_visible(False)
  axarr[0,2].yaxis.set_visible(False)


  img6 = axarr[1,2].imshow(rfb3_match_full, vmin=0,vmax=1)
  axarr[1,2].set_title('Aster band 3N')
  div6 = make_axes_locatable(axarr[1,2])
  cax6 = div6.append_axes("right", size="10%", pad=0.05)
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
  axarr[2,2].set_xlabel('Reflectance')
  axarr[2,2].set_ylabel('Cumulative %')
  axarr[2,2].legend(loc=5)
  
  fig.savefig('images/modis_aster_{}.png'.format(file_end))
  plt.close()
  aster_match = [rfb1_match_full[~edge_mask],rfb2_match_full[~edge_mask],rfb3_match_full[~edge_mask]]
  modis_bands = [pbands[3][~edge_mask],pbands[0][~edge_mask],pbands[1][~edge_mask], pbands[2][~edge_mask]]
  aster_test = calc.getBlueAster(aster_match,modis_bands, edge_mask, reflectance_b1.shape)
  misc.imsave('images/rgb_{}_aster.png'.format(file_end),np.round(aster_test*255).astype('uint8'))
  misc.imsave('images/rgb_{}_modis.png'.format(file_end),np.round(color_img*255).astype('uint8'))
  
