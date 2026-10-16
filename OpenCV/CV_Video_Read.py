import cv2

cap = cv2.VideoCapture(0) #opens the camera and (camera number) if multiple cameras in bracket

while True:
    ret,frame = cap.read() #ret -> is a boolean value and stores True if frame captured successfully and false if failed
                 #frame -> actual image in numpy array

    if ret == False:
        continue
        
    cv2.imshow("Camera Image",frame)
    key = cv2.waitKey(100) #waits for 100 seconds
    if key == ord('q'): #exit on pressing q
        break




cv2.waitKey(0) #Waits indefinitely for a key press
cv2.destroyAllWindows() #Closes all OpenCV windows.
cap.release() #releases camera resource for other programs use 