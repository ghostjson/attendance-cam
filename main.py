from core.core_face_identification import FaceIdentification
import cv2
from core import video
import json
import os



# with open('data/subjects.json', 'w') as s:
#     s.write(subjects)

#create a face identificatin object
fi = FaceIdentification()

#fetch_data
#fi.fetch_data()

#Add a new person
#video.addFace("Sachin")




attendance = []
#console
#console again
if __name__ == "__main__":
    
    entry = 'd'
    while(entry != 'e'):
        print("Enter 'a' for add a new person:")
        print("Enter 's' for start taking attendance:")
        print("Enter 'e' for end taking attendance:")

        entry = input()

        if entry == 'a':
            while True:
                print("Enter person's name:")
                name = input()
                video.addFace(name)
                print("Want to add more person?[y/n]")
                if input() != 'y':
                    break
            fi.fetch_data()
            exit()
            
        elif entry == 's':
            while True:
                try:
                    video.stream()
                    prediction = fi.predictimg("img/frame0.jpg")

                    #if prediction fails return to menu
                    if prediction == False:
                        break

                    if prediction in attendance:
                        print(prediction + " already in attendance list")
                    else:
                        attendance.append(prediction)

                finally:
                #end
                    if input() == 'e':
                        entry = 'e'
                        try:
                            os.remove("img/frame0.jpg")
                            break
                        except FileNotFoundError:
                            raise Exception('frame0 is not found!')
                            break
        elif entry == 'e':
            break
        else:
            print("Invalid input try again")


    print("\n\n")


    print("Present:")
    for i in attendance:
        print(i)

