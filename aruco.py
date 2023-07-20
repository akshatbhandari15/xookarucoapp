import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd




filename = "IMG_20230120_165052.jpg"
frame = cv2.imread(filename)



gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

print(ids)
plt.figure()
aruco_coordinates_X = []
aruco_coordinates_Y = []
plt.imshow(frame_markers, origin = "upper")
if ids is not None:
    for i in range(len(ids)):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "+", label = "id={0}".format(ids[i]))
    for i in range(0, ids.size):         
        x = (corners[i-1][0][0][0] + corners[i-1][0][1][0] + corners[i-1][0][2][0] + corners[i-1][0][3][0]) / 4
        y = (corners[i-1][0][0][1] + corners[i-1][0][1][1] + corners[i-1][0][2][1] + corners[i-1][0][3][1]) / 4
        aruco_coordinates_X.append(x)
        aruco_coordinates_Y.append(y)
        
"""for points in rejectedImgPoints:
    y = points[:, 0]
    x = points[:, 1]
    plt.plot(x, y, ".m-", linewidth = 1.)"""
aruco_coordinates_X = np.array(aruco_coordinates_X)
aruco_coordinates_Y = np.array(aruco_coordinates_Y)
ind = np.argsort(aruco_coordinates_X)
aruco_coordinates_X = np.take_along_axis(aruco_coordinates_X, ind, axis = 0)
aruco_coordinates_Y = np.take_along_axis(aruco_coordinates_Y, ind, axis = 0)

print(" X coordinates: ", aruco_coordinates_X)
print( " Y Coordinates: ", aruco_coordinates_Y)

plt.legend()
plt.show()


