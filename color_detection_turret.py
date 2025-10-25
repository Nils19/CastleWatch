#
#Developed by Vikrant Fernandes
#

import numpy as np
import cv2
import serial

#Capture from external USB webcam instead of the in-built webcam (shitty quality)
cap = cv2.VideoCapture(0)

#kernel window for morphological operations
kernel = np.ones((5,5),np.uint8)

#resize the capture window to 640 x 480
ret = cap.set(3,640)
ret = cap.set(4,480)

lower_color = np.array([5, 120, 150])
upper_color = np.array([35, 255, 255])


#begin capture
while(True):
    ret, frame = cap.read()

    #Smooth the frame
    frame = cv2.GaussianBlur(frame,(3,3),0)

    #Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Mask to extract just the red pixels
    mask = cv2.inRange(hsv,lower_color,upper_color)
    # mask2 = cv2.inRange(hsv,lower_red2,upper_red2)
    # mask = cv2.add(mask,mask2)

    #morphological opening
    mask = cv2.erode(mask,kernel,iterations=2)
    mask = cv2.dilate(mask,kernel,iterations=2)

    #morphological closing
    mask = cv2.dilate(mask,kernel,iterations=2)
    mask = cv2.erode(mask,kernel,iterations=2)

    #Detect contours from the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]

    if(len(cnts) > 0):
        #Contour with greatest area
        c = max(cnts,key=cv2.contourArea)
        
        # Calculate the center using moments
        M = cv2.moments(c)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Draw the calculated center point on mask
            cv2.circle(mask, (center_x, center_y), 5, 255, -1)
            
            #Radius and center pixel coordinate of the largest contour for the circle
            ((x,y),radius) = cv2.minEnclosingCircle(c)

            if radius > 5:
                #Draw an enclosing circle
                cv2.circle(frame,(int(x), int(y)), int(radius),(0, 255, 255), 2)
                #Draw the calculated center
                cv2.circle(frame,(center_x, center_y), 5, (255, 0, 0), -1)

                #Draw a line from the center of the frame to the center of the contour
                cv2.line(frame,(320,240),(center_x, center_y),(0, 0, 255), 1)
                #Reference line
                cv2.line(frame,(320,0),(320,480),(0,255,0),1)

                radius = int(radius)

                #distance of the 'x' coordinate from the center of the frame
                #wdith of frame is 640, hence 320
                length = 320-center_x

    #display the image
    cv2.imshow('frame',frame)
    #Mask image
    cv2.imshow('mask',mask)
    #Quit if user presses 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#Release the capture
cap.release()
cv2.destroyAllWindows()