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
    
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    coords = np.dstack((x,y,z))
    
    return coords

def sphereToDegree(coords, radius):
    x = coords[:,:,0]
    y = coords[:,:,1]
    z = coords[:,:,2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.rad2deg(np.arcsin(z/r))
    lon = np.rad2deg(np.arctan2(y,x))
    

    degree_coords = np.dstack((lat,lon))
    return degree_coords

def bilinearInterp(corners, h_step, v_step):
    """This function interpolates between the four
    corners and then fills in the middle of the surface
    
    only upper right corner is in returned surface
    
    cxxxxxxxc
    xxxxxxxx
    xxxxxxxx
    c       c
    """
    
    ul = corners[0,0]
    ur = corners[0,1]
    ll = corners[1,0]
    lr = corners[1,1]

    #surface = np.zeros((v_step,h_step))
    surface = []

    xt = np.linspace(ul,ur,num=h_step,endpoint=False)
    xb = np.linspace(ll,lr,num=h_step,endpoint=False)

    for i,j in zip(xt,xb):
        surface.append(np.linspace(i,j,num=v_step,endpoint=False))

    return np.column_stack(surface)

    """
    # interpolate top and bottom row
    for i in range(h_step):
        surface[0,i] = (1 - i/h_step)*ur + (i/h_step)*ul
        surface[v_step-1,i] = (1 - i/h_step)*lr + (i/h_step)*ll
    
    
    
    # interpolate cente
    for i in range(1,v_step-1):
        for j in range(h_step):
            xt = surface[0,j]
            xb = surface[v_step-1,j]
            surface[i,j] = (1 - i/v_step)*xt + (i/v_step) * xb
    """

def geoInterp(coords,h_step, v_step):
    """This function interpolates between the values
    found in coords

    Input
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
        for i in range(coords.shape[0]-1):
            for j in range(coords.shape[1]-1):
                ur = dim[i,j]
                ul = dim[i,j+1]
                lr = dim[i+1,j]
                ll = dim[i+1,j+1]
                corners = np.array(([ur,ul],[lr,ll]))
                surface = bilinearInterp(corners,h_step,v_step)
                full_coords[i*v_step:(i+1)*v_step,j*h_step:(j+1)*h_step,d] = surface

    return full_coords

