from core.core_face_identification import FaceIdentification
import cv2
from core import video

#subjects
subjects = ["Will","Mark","Jobs","Me"]

#create a face identificatin object
fi = FaceIdentification(subjects)

#fetch_data
#fi.fetch_data()

video.stream()


#predict the given image
#fi.predictimg("test_data/s1.jpg")



