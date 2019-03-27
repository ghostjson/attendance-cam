import cv2
import numpy as np
import os
import pickle


#face_identification class
class FaceIdentification:
    
    def __init__(self,subjects):
        self.subjects = subjects

    #function to detect face using OpenCV
    def detect_face(self,img):
        #convert the test image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #load OpenCV face detector
        face_cascade = cv2.CascadeClassifier('cascade/lbpcascade_frontalface.xml')
        
        #detect multiscale images
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        
        #if no faces are detected then return None
        if (len(faces) == 0):
            return None, None
        
        #extract the face area assuming there is only one face
        x, y, w, h = faces[0]
        
        #return only the face part of the image
        return gray[y:y+w, x:x+h], faces[0]

    #this function will read all persons' training images, detect face from each image
    #and will return two lists of exactly same size, one list 
    #of faces and another list of labels for each face
    def prepare_training_data(self,data_folder_path):
        
        #get the directories 
        dirs = os.listdir(data_folder_path)
        
        #list to hold all subject faces
        faces = []
        #list to hold labels for all subjects
        labels = []
        
        #get each image in dir
        for dir_name in dirs:
        
            #ignore any non-relevant directories if any
            if not dir_name.startswith("p"):
                continue;
            
            #get label
            label = int(dir_name.replace("p", ""))
            
            #build path of directory containing images for current subject subject
            subject_dir_path = data_folder_path + "/" + dir_name
            
            #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)


            #go through each image name, read image, 
            #detect face and add face to list of faces
            for image_name in subject_images_names:
            
                #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;
                
                #build image path
                #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                #read image
                image = cv2.imread(image_path)
                
                #display an image window to show the image 
                cv2.imshow("Training on image...", image)
                cv2.waitKey(100)
                
                #detect face
                face, rect = self.detect_face(image)
                
                
                #ignore faces that are not detected
                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    labels.append(label)
                    #print(image_name+" read sucessfully!!")
                else:
                    print(image_name+" can't detect face!")
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        

        return faces, labels

    def fetch_data(self):
        #fetch data
        print("Preparing data...")
        self.faces, self.labels = self.prepare_training_data("training_data")
        print("Data prepared")

        #print total faces and labels
        print("Total faces: ", len(self.faces))
        print("Total labels: ", len(self.labels))

    def trainData(self):
        #face recogizing algorithm 
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        #training algorithm
        self.face_recognizer.train(self.faces, np.array(self.labels))

        

    #function to draw rectangle on image 
    def draw_rectangle(self,img, rect):
        (self.x, self.y, self.w, self.h) = rect
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), (0, 255, 0), 2)
    
    #function to draw text on give image from x,y 
    def draw_text(self,img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


    #predict the image of the given person
    def predict(self,test_img):


        print("Predicting images...")
        #load test images
        img = test_img = cv2.imread(test_img)

        #detect face from the image
        face, rect = self.detect_face(img)

        #predict the image using our face recognizer 
        label = self.face_recognizer.predict(face)
        
        #get name of respective label returned by face recognizer
        label_text = self.subjects[label[0]]
        
        print("Accuracy:"+str(label[1]))

        #draw a rectangle around face detected
        self.draw_rectangle(img, rect)
        #draw name of predicted person
        self.draw_text(img, label_text, rect[0], rect[1]-5)
        
        print("Prediction complete")
        return img
        
    def predictimg(self,img):
        #perform a prediction
        predicted_img = self.predict(img)


        #display both images
        cv2.imshow("Face Prediction", predicted_img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


