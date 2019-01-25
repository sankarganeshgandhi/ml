import cv2
import sys

print('Press ESC to quit')

inputImgPath = sys.argv[1]
cascadeXMLPath = "./haarcascade_frontalface_default.xml"

faceFrontCascade = cv2.CascadeClassifier(cascadeXMLPath)

inputImg = cv2.imread(inputImgPath)
gray = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)

faces = faceFrontCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(inputImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Face found", inputImg)
cv2.waitKey(0)
