import cv2
import pyaudio
import wave
import threading
import os
import constants


def soundUserName(userName):
    CHUNK = 1024

    wf = wave.open(os.path.join(constants.FILEPATH_TO_SOUNDS_DIR, 'hello.wav'), 'rb')
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(CHUNK)

    # play stream (3)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # stop stream (4)
    stream.stop_stream()
    stream.close()
    wf.close()

    waveFile = os.path.join(constants.FILEPATH_TO_SOUNDS_DIR, userName + '.wav')

    wf = wave.open(waveFile, 'rb')
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(CHUNK)

    # play stream (3)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # stop stream (4)
    stream.stop_stream()
    stream.close()
    wf.close()


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(constants.FILEPATH_TO_TRAINER_YAML)
faceCascade = cv2.CascadeClassifier(constants.FILEPATH_TO_HAAR_CASCADE_XML);
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
index = 0

# names related to ids: example ==> 'UserName1': id=1,  etc
userNames = ['None', '1', '2', '3']

# Initialize and start realtime video capture
videoCam = cv2.VideoCapture(0)
videoCam.set(3, 640)  # set video widht
videoCam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * videoCam.get(3)
minH = 0.1 * videoCam.get(4)

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

userName = 'None'
quitVideo = False
lastUserName = 'None'
threads = []
while quitVideo is False:
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
            break
        else:
            userName = 'None'
            confidence = "  {0}%".format(round(100 - confidence))

    if userName is not 'None' and lastUserName is not userName:
        lastUserName = userName
        someThreadAlive = False
        for thread in threads:
            if thread.is_alive():
                someThreadAlive = True
                break
        if someThreadAlive is False:
            t = threading.Thread(target=soundUserName, args=(userName,))
            threads.append(t)
            t.start()

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

    cv2.imshow('Video', img)
    k = cv2.waitKey(30) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        quitVideo = True

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
videoCam.release()
cv2.destroyAllWindows()

# close PyAudio (5)
p.terminate()