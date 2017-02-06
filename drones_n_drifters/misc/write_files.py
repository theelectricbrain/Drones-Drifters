#!/usr/bin/python2.7
# encoding: utf-8

import numpy as np
import simplekml
from scipy.io import savemat
import os
from osgeo import ogr
from osgeo import osr
from utilities import *


def write_tracks2kml(tracksInDeg, kmlName):
    """
    Writes trajectories in KML file

    :param tracksInDeg: trajectories in degrees
    :param kmlName: KML file name
    """
    kml = simplekml.Kml()
    for ii, tr in enumerate(tracksInDeg):
        lin = kml.newlinestring(name="Trajectory "+str(ii), coords=tr)
    kml.save(kmlName)

    return


def write_contours2kml(surf_contours, kmlName, fill=True):
    """
    Writes contours into KML file

    :param surf_contours: contours in degrees
    :param kmlName: kml file name
    :param fill: fills contours' areas with solid color
    """

    kml = simplekml.Kml()
    for ii, contour in enumerate(surf_contours):
        contour.append(contour[0])
        pol = kml.newpolygon(name="Contour "+str(ii), outerboundaryis=contour)
        if not fill:
            pol.polystyle.fill = 0
    kml.save(kmlName)

    return


def write2drifter(d, uav, matName):
    """
    Writes oceanographic quantities to matfile

    :param d: dataframe
    :param uav: UAV object
    :param matName: mat filename
    """
    d4mat = {}
    d4mat['comments'] = ["trajectories from pumpkins"]
    d4mat['comments'].append("reference time: " + uav.timeRef.strftime("%Y-%m-%d %H:%M:%S"))
    u = []
    v = []
    lon = []
    lat = []
    times = []
    for key in d.keys():
        u.extend(d[key]['U'].tolist())
        v.extend(d[key]['V'].tolist())
        lon.extend(d[key]['longitude'].tolist())
        lat.extend(d[key]['latitude'].tolist())
        times.extend(datetime_to_mattime(d[key].index.tolist()))
    d4mat['velocity'] = {}
    d4mat['velocity']['u'] = np.asarray(u)
    d4mat['velocity']['v'] = np.asarray(v)
    d4mat['velocity']['vel_lon'] = np.asarray(lon)
    d4mat['velocity']['vel_lat'] = np.asarray(lat)
    d4mat['velocity']['vel_time'] = np.asarray(times)

    savemat(matName, d4mat)

    return

def write2shp(d, title):
    """
    Writes velocity components in shapefile

    :param d: dataframe
    :param title: shapefile's title
    """

    # # Reformat file name
    # title = title.replace(" ", "_")
    # title = title.replace("(", "_")
    # title = title.replace(")", "_")
    # title = title.replace("-", "_")
    # title = title.replace("/", "_")
    #
    if not title[-4:] == '.shp':
        filename = title + '.shp'
    else:
        filename= title

    # Reading from dataframe
    u = []
    v = []
    lon = []
    lat = []
    for key in d.keys():
        u.extend(d[key]['U'].tolist())
        v.extend(d[key]['V'].tolist())
        lon.extend(d[key]['longitude'].tolist())
        lat.extend(d[key]['latitude'].tolist())
    u = np.asarray(u)
    v = np.asarray(v)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

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
    lyr.CreateField(ogr.FieldDefn("U", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("V", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("Flow speed", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("Dir", ogr.OFTReal))

    # Change angle convention
    dir = np.rad2deg(np.arctan2(v, u))
    dir *= -1.0  # anti-C tp clockwise
    dir[np.where(dir < 0.0)] -= 360.0  # [-180, 180] to [0, 360]
    dir += 90.0  # true North
    dir[np.where(dir >= 360.0)] -= 360.0

    for ii in range(lon.shape[0]):
        # create the feature
        feature = ogr.Feature(lyr.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("U", u[ii])
        feature.SetField("V", v[ii])
        feature.SetField("Flow Speed", np.sqrt(u[ii]**2.0 + v[ii]**2.0))
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

# TODO: def write_contours2shp(surf_contours, fill=True)