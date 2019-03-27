import cv2



def stream():
    flag = False
    #get cam video object
    cap = cv2.VideoCapture(0)

    #count = 1

    while True:

        _,frame = cap.read()   #get each frame

        cv2.imshow('Video',frame) #show each frame

        if not flag:

            name = "img/frame.jpg"
            cv2.imwrite(name,frame)
            flag = True
        #break when esc key is pressed
        k = cv2.waitKey(1)
        if k == 27:
            break

        #if count > 5:
        #	break

    cv2.destroyAllWindows()
    cap.release() #release all resources

