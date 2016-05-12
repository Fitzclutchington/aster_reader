from osgeo import gdal, osr
from pyproj import Proj
import numpy as np

ds = gdal.Open('data1.l3a.vnir1.tif')
x0, dx, _, y0, _, dy = ds.GetGeoTransform()
w = x0
e = x0 + ds.RasterXSize * dx
n = y0
s = y0 + ds.RasterYSize * dy

ty,tx = np.mgrid[n:s:dy,w:e:dx]

srs = osr.SpatialReference()
srs.ImportFromWkt(ds.GetProjection())
projstr = srs.ExportToProj4()

b1 = ds.GetRasterBand
meta = open("data1.l3a.gh").read()
meta = meta.replace("\n",'')
meta = meta.split(';')
meta_dict = [x.split('=') for x in meta]
gain_info = meta_dict["Gain"].strip(')}').strip('{(')
gain_info.split('),(')
gain_dict = dict([x.replace("'",'').split(',') for x in gain_info])