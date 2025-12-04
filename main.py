import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]
emojis = [
    cv2.imread(f'emoji/{emotion}.png', cv2.IMREAD_UNCHANGED)
       for emotion in emotion_labels
]

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    if len(faces) == 0:
        continue
    (x, y, w, h) = sorted(faces, key=lambda face: face[2] * face[3])[-1]
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        emoji = cv2.resize(emojis[prediction.argmax()], (w, h))
        for c in range(0, 3):
            frame[y:y + h, x:x + w, c] = emoji[:, :, c] * (emoji[:, :, 3] / 255.0) + frame[y:y + h, x:x + w, c] * (
                        1.0 - emoji[:, :, 3] / 255.0)
        label_position = (x, y)
        # cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()