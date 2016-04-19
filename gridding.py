import numpy as np

def degreeToSphere(lat,lon,radius):
    """This function converts latitude and longitude into
    x,y,z coordinates.

    lat, lon: 11 x 11 numpy arrays of latitude and longitude
              in degrees
    radius : radius of the earth at the given granule

    Returns
    -------
    coords: 11 x 11 x 3 numpy array of x,y,z coordinates"""

    x = radius * np.cos(lon) * np.sin(lat)
    y = radius * np.sin(lon) * np.sin(lat)
    z = radius * np.cos(lon)

    coords = np.dstack((x,y,z))

    return coords

def sphereToDegree(coords, radius):
    x = coords[:,:,0]
    y = coords[:,:,1]
    z = coords[:,:,2]
    lon = np.arccos(z/radius)
    lat = np.arctan(np.divide(y,x))
    degree_coords = np.dstack((lat,lon))
    return degree_coords

def geoInterp(coords,h_step, v_step):
    """This function interpolates between the values
    found in coords

    Inout
    ------
    coords: 11 x 11 x 3 numpy array of x,y,z coordinates
    h_step: horizontal step between coordinates
    v_step: vertical step between coordinates

    output
    ------
    full_coords: 4200 x 4920 x3 numpy array containing
                 interpolated x,y,z values
    """
    #note numpy shape is (h,w,d)
    full_coords = np.zeros((v_step*10,h_step*10,3))
    for d in range(coords.shape[2]):
        dim = coords[:,:,d]

