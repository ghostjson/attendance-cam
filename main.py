from core.core_face_identification import FaceIdentification

#subjects
subjects = ["Will","Mark"]

#create a face identificatin object
fi = FaceIdentification(subjects)
#fetch_data
fi.fetch_data()
#train that fetched data
fi.trainData()


#predict the given image
fi.predictimg("test_data/3.jpg")
