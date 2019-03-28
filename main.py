from core.core_face_identification import FaceIdentification
import cv2
from core import video
import json

#subjects
# subjects = ["Will","Mark","Jobs","Me"]
# subjects = json.dumps(subjects, indent=4)

# with open('data/subjects.json', 'w') as s:
#     s.write(subjects)

#create a face identificatin object
fi = FaceIdentification()

#fetch_data
#fi.fetch_data()

#get sample frames from webcame to predict
video.stream()

#Add a new person
#video.addFace("Me")

#FaceIdentification.addPerson("hello")

#predict the given image
prediction = []
for i in range(0,16, 5):
    prediction.append(fi.predictimg("img/frame%d"%i + ".jpg"))


#print(fi.predictimg('img/frame0.jpg'))
print(max(set(prediction), key=prediction.count))



