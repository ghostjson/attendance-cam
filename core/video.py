import cv2
from core.core_face_identification import FaceIdentification


def stream():
    flag = False
    #get cam video object
    cap = cv2.VideoCapture(0)

    #count = 1

    while True:

        _,frame = cap.read()   #get each frame

        
        #detect face in video
        face = FaceIdentification.detect_face(frame)
        
        #draw rec on the frame if detect
        try:
            face = face[1]
            x,y,w,h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Video',frame) #show each frame
            
        except:
            cv2.imshow('Video',frame) #show each frame

        #break when esc key is pressed
        k = cv2.waitKey(1)
        if k == 27:
            break

        #if count > 5:
        #	break

    cv2.destroyAllWindows()
    cap.release() #release all resources


        # if not flag:

            # name = "img/frame.jpg"
            # cv2.imwrite(name,frame)
            # flag = True