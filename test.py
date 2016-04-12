from pyhdf import SD
hdf = SD.SD('AST_L1B_00306072015231418_20150608115103_32348.hdf')
hdf.attributes()
hdf.attributes
hdf.datasets()
hdf.select('ImageData1')
b1 = hdf.select('ImageData1')
db1 = b1.get()
type(db1)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(db1)
plt.colorbar()
plt.show()
%hist
hdf.info()
hdf.nametoindex()
hdf.attr
hdf.attr()
hdf.attributes()
hdf.datasets()
hdf.attributes()
hdf.attributes().keys()
hdf.attributes('productmetadata.0')
hdf.attributes('productmetadata.0').keys()
hdf.attributes()['productmetadata.0']
hdf.attributes()['productmetadata.0'].keys()
hdf.attributes()['productmetadata.0']['SOLARDIRECTION']
hdf.attributes()['productmetadata.0'][1]
pm0 = hdf.attributes()['productmetadata.0']
pm0 = [p.strip() for p in pm0.split('\n')]
pm0[0]
pm0[1]
pm0[2]
pm0[3]
pm0[4]
pm0 = hdf.attributes()['productmetadata.0']
cm0 = [(p.split('=')[0].strip(),p.split('=')[1].strip()) for p in pm0.split('\n') if p != '']
pm0 = hdf.attributes()['productmetadata.0']
cm0 = [p.strip() for p in pm0.split('\n') if p != '']
cm0[:10]
cm0 = [l.split('=') for l in cm0]
cm0[:10]
sdirect = [l for l in cm0 if l[0] == 'SOLARDIRECTION']
sdirect
sdirect = [l for l in cm0 if l[0].strip() == 'SOLARDIRECTION']
sdirect
sdind = cm0.find(['OBJECT','SOLARDIRECTION'])
sdind = cm0.index(['OBJECT','SOLARDIRECTION'])
len(cm)
len(cm0)
cm0
%hist
from pyhdf import odl_parser
import pyhdf
%hist
cm0
hdf.datasets()
import odl_parser
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
meta
import odl_parser
reload("odl_parser")
reload(odl_parser)
reload(odl_parser)
reload(odl_parser)
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
reload(odl_parser)
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
type([]) == 'list'
type([])
type([]) == list
reload(odl_parser)
reload(odl_parser)
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
reload(odl_parser)
meta = odl_parser.parsemeta(hdf.attributes()['productmetadata.0'])
meta
meta.keys()
meta.keys().keys()
meta.keys()
meta['ASTERGENERICMETADATA']
meta['ASTERGENERICMETADATA'].keys()
