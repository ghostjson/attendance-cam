import cv2
from core.core_face_identification import FaceIdentification

#get sample frames from webcame to predict
def stream():
    
    #count = 0
    #get cam video object
    cap = cv2.VideoCapture(0)

    while True:

        _,frame = cap.read()   #get each frame

        
        #detect face in video
        face = FaceIdentification.detect_face(frame)
        
        #draw rec on the frame if detect
        try:
            face = face[1]
            x,y,w,h = face

            
            #save photos
        #    if count % 5 == 0:
            name = "img/frame0.jpg"
            cv2.imwrite(name,frame)
        #    count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Video',frame) #show each frame
            break

        except:
            cv2.imshow('Video',frame) #show each frame

        #break when esc key is pressed
        k = cv2.waitKey(1)
        if k == 27:
            return False
            break

        #if count > 15:
        #	break
        #break
    cv2.destroyAllWindows()
    cap.release() #release all resources


        


def addFace(name):
    #get cam video object
    cap = cv2.VideoCapture(0)

    #add person name and return persons label
    label = FaceIdentification.addPerson(name)
    count = 0

    while True:

        _,frame = cap.read()   #get each frame

        
        #detect face in video
        face = FaceIdentification.detect_face(frame)
        
        #draw rec on the frame if detect
        try:
            face = face[1]
            x,y,w,h = face
            

            #save photos
            if count % 5 == 0:
                name = "training_data/p"+str(label)+"/frame%d.jpg"%count
                cv2.imwrite(name,frame)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Video',frame) #show each frame
            
        except:
            cv2.imshow('Video',frame) #show each frame

        #break when esc key is pressed
        k = cv2.waitKey(1)
        if k == 27:
            break

        if count > 100:
        	break

    cv2.destroyAllWindows()
    cap.release() #release all resources