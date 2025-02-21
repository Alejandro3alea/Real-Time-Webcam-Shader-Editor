import pyvirtualcam
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

webcam = cv2.VideoCapture(0)

video = cv2.VideoCapture('max300.mp4')

camWidth = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
camHeight = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

segmentor = SelfiSegmentation()


looping = True
with pyvirtualcam.Camera(width=camWidth, height=camHeight, fps=20) as outCam:
    while looping:

        success, frame = webcam.read()
        success2, videoFrame = video.read()

        videoFrame = cv2.resize(videoFrame, (camWidth, camHeight), interpolation=cv2.INTER_AREA)

        blurredFrame = cv2.blur(frame, (10, 10))
        segmentatedImage = segmentor.removeBG(frame, videoFrame, 0.8)

        concatenatedImage = cv2.hconcat([frame, segmentatedImage])

        cv2.imshow("Real Time Webcam Shader Editor", concatenatedImage)

        outCam.send(frame)
        outCam.sleep_until_next_frame()

        if cv2.waitKey(1) == ord('q'):
            looping = False


webcam.release()
cv2.destroyAllWindows()