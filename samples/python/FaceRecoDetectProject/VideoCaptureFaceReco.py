import cv2

videoCap = cv2.VideoCapture(0)

# setting the width and height of the camera image capture
videoCap.set(3, 640)
videoCap.set(4, 480)

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    ret, frame = videoCap.read()
    frame = cv2.flip(frame, 1)  # Camera in the right position
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('Video', frame)

    ##
    # cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    ##

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

videoCap.release()
cv2.destroyAllWindows()
