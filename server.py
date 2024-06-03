from flask import Flask, request, jsonify
from PIL import Image
import io
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_50epochs.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/upload', methods=['POST'])
def upload():
    try:
        image_data = request.data
        image = Image.open(io.BytesIO(image_data))
        
        # Преобразование изображения типа Image в массив numpy для использования с cv2
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        emotion_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_model.predict(roi)[0]
            if preds.argmax() == 5:  # если улыбка распознана
                emotion_detected = True
                break

        if emotion_detected:
            result = {'result': True}
        else:
            result = {'result': False}
        
        return jsonify(result)  # Отправляем результат в виде JSON-ответа
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
