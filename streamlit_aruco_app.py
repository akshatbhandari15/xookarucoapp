import cv2
import numpy as np
import streamlit as st
import cv2, PIL
from cv2 import aruco
from scipy.spatial import distance

uploaded_file = st.file_uploader("Choose a image file", type="jpg")
mtx = np.load("calibration_matrix.npy")
dist = np.load("distortion_coefficients.npy")
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)    
    print("Loading image...")
    #if ret returns false, there is likely a problem with the webcam/camera.
    #In that case uncomment the below line, which will replace the empty frame 
    #with a test image used in the opencv docs for aruco at https://www.docs.opencv.org/4.5.3/singlemarkersoriginal.jpg
    # frame = cv2.imread('./images/test image.jpg') 

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 10
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(corners)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #get pixel mm ratio
    int_corners = np.int0(corners)
    cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)
    aruco_perimeter = 0.0
    for i in range(len(ids)):
        aruco_perimeter += cv2.arcLength(corners[i], True)
    aruco_perimeter = aruco_perimeter/len(ids)    
    #print(aruco_perimeter)
    pixel_mm_ratio = aruco_perimeter / 256
    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec , objPoints = aruco.estimatePoseSingleMarkers(corners, 0.064, mtx, dist)
        #print(objPoints.shape)
        #print(objPoints)


        #(rvec-tvec).any() # get rid of that nasty numpy value array error
        
        strg = ''
        for i in range(0, ids.size):
            # draw axis for the aruco markers
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], 0.1)
            imagePoints, jacobian = cv2.projectPoints(objPoints, rvec[i], tvec[i], mtx, dist)
            print(imagePoints.squeeze().shape)
            #print(imagePoints)                
            #np.linalg.norm(tvec1-tvec2)
            imagePoints = imagePoints.squeeze()
            #print(imagePoints)

            
            x = (imagePoints[0][0] + imagePoints[1][0] + imagePoints[2][0] + imagePoints[3][0]) / 4

            y = (imagePoints[0][1] + imagePoints[1][1] + imagePoints[2][1] + imagePoints[3][1]) / 4

            cv2.putText(frame, " Id: " + str(ids[i][0]), (int(x)+20, int(y)+20), font, 1, (0,255,0),2, cv2.LINE_AA)
            strg += ' [ ' + str(ids[i][0])+ '  x: ' + str(x) + ' y: ' + str(y) + '] '
            


        for i in range(len(ids)):
            for j in range(len(ids)):
                imagePoints, jacobian = cv2.projectPoints(objPoints, rvec[i], tvec[i], mtx, dist)
                imagePoints = imagePoints.squeeze()

                x1 = (imagePoints[0][0] + imagePoints[1][0] + imagePoints[2][0] + imagePoints[3][0]) / 4
                
                y1 = (imagePoints[0][1] + imagePoints[1][1] + imagePoints[2][1] + imagePoints[3][1]) / 4

                imagePoints, jacobian = cv2.projectPoints(objPoints, rvec[j], tvec[j], mtx, dist)
                imagePoints = imagePoints.squeeze()
                x2 = (imagePoints[0][0] + imagePoints[1][0] + imagePoints[2][0] + imagePoints[3][0]) / 4
                
                y2 = (imagePoints[0][1] + imagePoints[1][1] + imagePoints[2][1] + imagePoints[3][1]) / 4

                #print(x1, y1, x2, y2)

                distan = distance.euclidean([x1, y1], [x2, y2])
                #print(distan)
                distan = float(distan)/pixel_mm_ratio

                print(" Distance between " + str(ids[i]) + " and " + str(ids[j]) + " is " + str(distan) + (" mm "))        
            
            

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)
        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame
    # Now do something with the image! For example, let's display it:
    st.image(frame, channels="BGR")