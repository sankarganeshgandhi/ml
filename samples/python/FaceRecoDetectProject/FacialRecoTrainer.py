import cv2
import numpy
from PIL import Image
import os
import constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
haarCascadeDetector = cv2.CascadeClassifier(constants.FILEPATH_TO_HAAR_CASCADE_XML)

def getImagesAndLabels(dirPath):
    imagesPath = [os.path.join(dirPath, f) for f in os.listdir(dirPath)]
    faceSamples = []
    ids = []
    for imagePath in imagesPath:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = numpy.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = haarCascadeDetector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


print('\n [INFO] Training faces. It will take a few seconds. Wait ...')

faces, ids = getImagesAndLabels(constants.FILEPATH_TO_DATASET)
recognizer.train(faces, numpy.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write(constants.FILEPATH_TO_TRAINER_YAML)

# Print the numer of faces trained and end program
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(numpy.unique(ids))))
