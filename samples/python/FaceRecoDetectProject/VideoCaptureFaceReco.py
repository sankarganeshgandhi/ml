import cv2
import constants

print('##### Press ESC to quit when video capture is on #####')

userId = input('Enter User Id: ')

print("\n[INFO] Initializing face capture. Look at the camera and wait ...")

videoCam = cv2.VideoCapture(0)

# setting the width and height of the camera image capture
videoCam.set(3, 640)
videoCam.set(4, 480)

faceCascade = cv2.CascadeClassifier(constants.FILEPATH_TO_HAAR_CASCADE_XML)

count = 0
while True:
    ret, frame = videoCam.read()
    frame = cv2.flip(frame, 1)  # Camera in the right position
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=1,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite("./dataset/User." + str(userId) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])

    frameTitle = 'Capturing images of {0} (<ESC> to Quit)'.format(userId)
    cv2.imshow(frameTitle, frame)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
    elif count >= constants.IMG_CAPTURE_COUNT:
        break

print('\n [INFO] Exiting Program and cleanup stuff')
videoCam.release()
cv2.destroyAllWindows()
