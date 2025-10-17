import cv2 as cv
import imutils

hog = cv.HOGDescriptor()

hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

capture = cv.VideoCapture('Src/people.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        (regions,_) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in regions:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow("Frame", frame)
        if cv.waitKey(25) & 0xff == ord('q'):
            break

    else:
        break

capture.release()
cv.destroyAllWindows()