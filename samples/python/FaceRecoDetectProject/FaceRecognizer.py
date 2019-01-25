import cv2
import constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(constants.FILEPATH_TO_TRAINER_YAML)
faceCascade = cv2.CascadeClassifier(constants.FILEPATH_TO_HAAR_CASCADE_XML);
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
index = 0

# names related to ids: example ==> Marcelo: id=1,  etc
userNames = ['None', 'Sankar', 'Sudha', 'Puppy']

# Initialize and start realtime video capture
videoCam = cv2.VideoCapture(0)
videoCam.set(3, 640)  # set video widht
videoCam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * videoCam.get(3)
minH = 0.1 * videoCam.get(4)

while True:
    ret, img = videoCam.read()
    img = cv2.flip(img, 1)  #
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        index, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # If confidence is less them 100 ==> "0" : perfect match
        if confidence < 100:
            userName = userNames[index]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            userName = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            userName,
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x+5, y+h-5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
videoCam.release()
cv2.destroyAllWindows()