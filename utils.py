import sys
from calendar import monthrange
import odl_parser
import os
from pyhdf import SD
import pandas as pd
import numpy as np

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

def openHDF(filename):
  fil, file_extension = os.path.splitext(filename)

  if file_extension != '.hdf':
    print "Not an HDF file"
    sys.exit(1)

  hdf = SD.SD(filename)

  return hdf