import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/stress_detector.h5')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = ['Low', 'Medium', 'High']
colors = [(0,255,0), (0,255,255), (0,0,255)]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.reshape(1,48,48,1)/255.0

        pred = model.predict(face)
        idx = np.argmax(pred)

        cv2.rectangle(frame, (x,y), (x+w,y+h), colors[idx], 2)
        cv2.putText(frame, labels[idx], (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[idx], 2)

    cv2.imshow("Stress Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
