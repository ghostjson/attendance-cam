from core.core_face_identification import FaceIdentification

#subjects
subjects = ["Will","Mark","Jobs"]

#create a face identificatin object
fi = FaceIdentification(subjects)

#fetch_data
#fi.fetch_data()



#predict the given image
fi.predictimg("test_data/s2.jpg")
