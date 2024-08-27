import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

while True:
  ret,frame = video_capture.read()
  # convert to grayscale
  frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
  detections = face_detector.detectMultiScale(frame_gray)
  #construct rectangle
  for(x,y,w,h) in detections:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
  # show result
  cv2.imshow('Video',frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
video_capture.release()
cv2.destroyAllWindows()