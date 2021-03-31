import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
    ret , img = cap.read()    
    gray_frame = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    

    faces = face_cascade.detectMultiScale(gray_frame , 1.1 , 5)

    for (x , y , w , h) in faces:
        cv2.rectangle(img , (x,y) , (x+w , y+h) , (255 , 0 , 0) , 3)

    cv2.imshow("Video Frame", img)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()