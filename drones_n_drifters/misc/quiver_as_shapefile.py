#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import numpy as np
from pyseidon_dvt import FVCOM
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import os
import zipfile
import scipy.interpolate as interpolate
from osgeo import ogr
from osgeo import osr

### HEADER
# Global variables
path2file = '/EcoII/Luna/simulations/2016-08-01_2016-08-31/output/'
filename = 'acadia_force_2d_2016-08-01_2016-08-16.nc'
bbOverview = [-64.462113, -64.399945,  45.351514,  45.380683]
bbZoom = [-64.445695, -64.416041,  45.358523,  45.373739]
startDate = "2016-08-08T09:00:00.00"
numberOfHours = 15
### END HEADER

### PROCESS
def save_map_as_shapefile_points(ua, va, x, y, title=' ', varLabel=' ', debug=False):
    """
    Saves map as shapefile

    Inputs:
      - var = gridded variable, 1 D numpy array (nele or nnode)
      - x = coordinates, 1 D numpy array (nele or nnode)
      - y = coordinates, 1 D numpy array (nele or nnode)

    Options:
      - title = file name, string
      - kwargs = keyword options associated with ???
    """

    if title == ' ':
        title = 'save_map_data'
    else:  # reformat file name
        title = title.replace(" ", "_")
        title = title.replace("(", "_")
        title = title.replace(")", "_")
        title = title.replace("-", "_")
        title = title.replace("/", "_")
        title = title.replace(".", "_")

    filename = title + '.shp'

    # Projection
    # epsg_in=4326
    epsg_in = 3857  # Google Projection

    # give alternative file name is already exists
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(filename):
        filename = filename[:-4] + "_bis.shp"

    shapeData = driver.CreateDataSource(filename)

    spatialRefi = osr.SpatialReference()
    spatialRefi.ImportFromEPSG(epsg_in)

    lyr = shapeData.CreateLayer("points_layer", spatialRefi, ogr.wkbPoint)

    # Features
    # Add the fields we're interested in
    #lyr.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
    #lyr.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("U", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("V", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("Flow speed", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("Dir", ogr.OFTReal))

    if debug: print "Writing ESRI Shapefile %s..." % filename
    lon = x[:]
    lat = y[:]

    # Change angle convention
    dir = np.rad2deg(np.arctan2(va, ua))
    dir *= -1.0  # anti-C tp clockwise
    dir[np.where(dir < 0.0)] -= 360.0  # [-180, 180] to [0, 360]
    dir += 90.0  # true North
    dir[np.where(dir >= 360.0)] -= 360.0

    for ii in range(lon.shape[0]):
        # create the feature
        feature = ogr.Feature(lyr.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("U", ua[ii])
        feature.SetField("V", va[ii])
        feature.SetField("Flow Speed", np.sqrt(ua[ii]**2.0 + va[ii]**2.0))
        feature.SetField("Dir", dir[ii])

        # create the WKT for the feature using Python string formatting
        wkt = "POINT(%f %f)" % (float(lon[ii]), float(lat[ii]))

        # Create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # Create the feature in the layer (shapefile)
        lyr.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()

    # Destroy the data source to free resources
    shapeData.Destroy()

    return

#  Kml header
kml_groundoverlay = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.0">
<Document>
<GroundOverlay>
  <name>__NAME__</name>
  <color>__COLOR__</color>
  <visibility>__VISIBILITY__</visibility>
  <Icon>
    <href>overlay.png</href>
  </Icon>
  <LatLonBox>
    <south>__SOUTH__</south>
    <north>__NORTH__</north>
    <west>__WEST__</west>
    <east>__EAST__</east>
  </LatLonBox>
</GroundOverlay>
<ScreenOverlay>
    <name>Legend</name>
    <Icon>
        <href>legend.png</href>
    </Icon>
    <overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>
    <screenXY x="0.015" y="0.075" xunits="fraction" yunits="fraction"/>
    <rotationXY x="0.5" y="0.5" xunits="fraction" yunits="fraction"/>
    <size x="0" y="0" xunits="pixels" yunits="pixels"/>
</ScreenOverlay>
</Document>
</kml>
'''


# Loading data and compute
fvcom = FVCOM(path2file+filename)
fvcom.Util2D.hori_velo_norm()
#fvcom.Util2D.vorticity()

# Find starting time index
for tt in range(fvcom.Variables.julianTime.shape[0]):
    timeText = ''.join(fvcom.Data.variables['Times'][tt])
    if startDate in timeText:
        startIndex = tt
        break

# Loop over hours
step = 6 # assuming 10 minutes time steps and hourly average
startI = 0 + startIndex
endI = step + startIndex

pbar = ProgressBar()
for ii in pbar(range(numberOfHours)):  #fvcom.Variables.julianTime.shape[0]//6)):
    # Average & Interpolation
    #vortR = 1.0 / np.abs(np.nanmean(fvcom.Variables.depth_av_vorticity[startI:endI, :], axis=0))
    flowspeed = np.nanmean(fvcom.Variables.hori_velo_norm[startI:endI, :], axis=0)
    ua = np.nanmean(fvcom.Variables.ua[startI:endI, :], axis=0)
    va = np.nanmean(fvcom.Variables.va[startI:endI, :], axis=0)

    # shapefiles
    name = 'flow speed and orientation ' + \
           ''.join(fvcom.Data.variables['Times'][startI]) + " and " + \
           ''.join(fvcom.Data.variables['Times'][endI])
    save_map_as_shapefile_points(ua, va, fvcom.Grid.lonc, fvcom.Grid.latc, title=name)

    #  Overview
    xf, yf = np.meshgrid(np.arange(bbOverview[0], bbOverview[1], np.abs((bbOverview[1] - bbOverview[0]) / 50.0)),
                         np.arange(bbOverview[2], bbOverview[3], np.abs((bbOverview[3] - bbOverview[2]) / 50.0)))
    orig = np.zeros((fvcom.Grid.lonc.shape[0], 2))
    orig[:, 0] = fvcom.Grid.lonc[:]
    orig[:, 1] = fvcom.Grid.latc[:]
    ask = np.zeros((xf.flatten().shape[0], 2))
    ask[:, 0] = xf.flatten()
    ask[:, 1] = yf.flatten()
    #interpol = interpolate.LinearNDInterpolator(orig, vortR)
    #vortRI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, ua)
    uaI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, va)
    vaI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, flowspeed)
    speedI = interpol(ask).reshape(xf.shape)
    #NvortRI = (vortRI/np.nanmax(vortRI))
    # Masking
    #indexM = np.where(np.isnan(NvortRI))
    #uaI[indexM] = np.nan
    #vaI[indexM] = np.nan
    ### END PROCESS

    ### PLOT
    # Initialize figure
    name = 'flow speed between ' + \
           ''.join(fvcom.Data.variables['Times'][startI]) + " and " + \
           ''.join(fvcom.Data.variables['Times'][endI])
    color = '9effffff'
    visibility = str(1)
    kmzfile = name+'.kmz'
    pixels = 2048  # pixels of the max. dimension
    units = 'm/s'
    geo_aspect = (1.0/np.cos(np.mean(fvcom.Grid.lat)*np.pi/180.0))
    xsize = np.abs(bbOverview[1] - bbOverview[0]) * geo_aspect
    ysize = np.abs(bbOverview[3] - bbOverview[2])
    aspect = ysize/xsize
    if aspect > 1.0:
        figsize = (30.0/aspect, 30.0)
    else:
        figsize = (30.0, 30.0*aspect)
    plt.ioff()
    fig = plt.figure(figsize=figsize, facecolor=None, frameon=False, dpi=pixels // 5)
    # fig = figure(facecolor=None, frameon=False, dpi=pixels//10)
    ax = fig.add_axes([0, 0, 1, 1])
    #pc = ax.quiver(xf, yf, uaI, vaI, speedI, cmap=plt.cm.jet, scale=1 / 0.015)
    pc = ax.quiver(xf, yf, uaI/speedI, vaI/speedI, speedI, cmap=plt.cm.jet, clim=[0,6.0], scale=1 / 0.015)
    #pc = ax.quiver(xf, yf, uaI*NvortRI, vaI*NvortRI, speedI, cmap=plt.cm.jet)  # weighted by vorticity
    ax.set_xlim([bbOverview[0], bbOverview[1]])
    ax.set_ylim([bbOverview[2], bbOverview[3]])
    ax.set_axis_off()
    fig.savefig('overlay.png', dpi=200, transparent=True)
    # Write kmz
    fz = zipfile.ZipFile(kmzfile, 'w')
    fz.writestr(name+'.kml', kml_groundoverlay.replace('__NAME__', name)\
                                                  .replace('__COLOR__', color)\
                                                  .replace('__VISIBILITY__', visibility)\
                                                  .replace('__SOUTH__', str(bbOverview[2]))\
                                                  .replace('__NORTH__', str(bbOverview[3]))\
                                                  .replace('__EAST__', str(bbOverview[1]))\
                                                  .replace('__WEST__', str(bbOverview[0])))
    fz.write('overlay.png')
    os.remove('overlay.png')

    # colorbar png
    fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)
    ax = fig.add_axes([0.0, 0.05, 0.2, 0.9])
    cb = fig.colorbar(pc, cax=ax)
    cb.set_label(units, color='0.0')
    for lab in cb.ax.get_yticklabels():
        plt.setp(lab, 'color', '0.0')

    fig.savefig('legend.png', transparent=True)
    fz.write('legend.png')
    os.remove('legend.png')
    fz.close()

    #  Zoom
    xf, yf = np.meshgrid(np.arange(bbZoom[0], bbZoom[1], np.abs((bbZoom[1] - bbZoom[0]) / 50.0)),
                         np.arange(bbZoom[2], bbZoom[3], np.abs((bbZoom[3] - bbZoom[2]) / 50.0)))
    orig = np.zeros((fvcom.Grid.lonc.shape[0], 2))
    orig[:, 0] = fvcom.Grid.lonc[:]
    orig[:, 1] = fvcom.Grid.latc[:]
    ask = np.zeros((xf.flatten().shape[0], 2))
    ask[:, 0] = xf.flatten()
    ask[:, 1] = yf.flatten()
    #interpol = interpolate.LinearNDInterpolator(orig, vortR)
    #vortRI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, ua)
    uaI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, va)
    vaI = interpol(ask).reshape(xf.shape)
    interpol = interpolate.LinearNDInterpolator(orig, flowspeed)
    speedI = interpol(ask).reshape(xf.shape)
    #NvortRI = (vortRI / np.nanmax(vortRI))
    # Masking
    #indexM = np.where(np.isnan(NvortRI))
    #uaI[indexM] = np.nan
    #vaI[indexM] = np.nan
    # Initialize figure
    name += "_zoom"
    color = '9effffff'
    visibility = str(1)
    kmzfile = name + '.kmz'
    pixels = 2048  # pixels of the max. dimension
    units = 'm/s'
    geo_aspect = (1.0 / np.cos(np.mean(fvcom.Grid.lat) * np.pi / 180.0))
    xsize = np.abs(bbZoom[1] - bbZoom[0]) * geo_aspect
    ysize = np.abs(bbZoom[3] - bbZoom[2])
    aspect = ysize / xsize
    if aspect > 1.0:
        figsize = (30.0 / aspect, 30.0)
    else:
        figsize = (30.0, 30.0 * aspect)
    plt.ioff()
    fig = plt.figure(figsize=figsize, facecolor=None, frameon=False, dpi=pixels // 5)
    # fig = figure(facecolor=None, frameon=False, dpi=pixels//10)
    ax = fig.add_axes([0, 0, 1, 1])
    #pc = ax.quiver(xf, yf, uaI, vaI, speedI, cmap=plt.cm.jet, scale=1 / 0.015)
    pc = ax.quiver(xf, yf, uaI/speedI, vaI/speedI, speedI, cmap=plt.cm.jet, clim=[0,6.0], scale=1 / 0.015)
    # pc = ax.quiver(xf, yf, uaI*NvortRI, vaI*NvortRI, speedI, cmap=plt.cm.jet)  # weighted by vorticity
    ax.set_xlim([bbZoom[0], bbZoom[1]])
    ax.set_ylim([bbZoom[2], bbZoom[3]])
    ax.set_axis_off()
    fig.savefig('overlay.png', dpi=200, transparent=True)
    # Write kmz
    fz = zipfile.ZipFile(kmzfile, 'w')
    fz.writestr(name + '.kml', kml_groundoverlay.replace('__NAME__', name) \
                .replace('__COLOR__', color) \
                .replace('__VISIBILITY__', visibility) \
                .replace('__SOUTH__', str(bbZoom[2])) \
                .replace('__NORTH__', str(bbZoom[3])) \
                .replace('__EAST__', str(bbZoom[1])) \
                .replace('__WEST__', str(bbZoom[0])))
    fz.write('overlay.png')
    os.remove('overlay.png')

    # colorbar png
    fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)
    ax = fig.add_axes([0.0, 0.05, 0.2, 0.9])
    cb = fig.colorbar(pc, cax=ax)
    cb.set_label(units, color='0.0')
    for lab in cb.ax.get_yticklabels():
        plt.setp(lab, 'color', '0.0')

    fig.savefig('legend.png', transparent=True)
    fz.write('legend.png')
    os.remove('legend.png')
    fz.close()

    # increment time bounds
    startI = endI
    endI += step
    #print "startI", startI
    #print "endI", endI
    #



