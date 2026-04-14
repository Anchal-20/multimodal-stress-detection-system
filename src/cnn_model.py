import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_dataset(data_dir):
    images, labels = [], []

    emotion_labels = {
        'angry': 0, 'disgust': 1, 'fear': 2,
        'happy': 3, 'sad': 4, 'surprise': 5,
        'neutral': 6
    }

    for emotion, label in emotion_labels.items():
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


def preprocess(X, y):
    stress_map = np.array([2, 1, 2, 0, 2, 1, 0])
    y = stress_map[y]

    X = X.reshape(-1, 48, 48, 1) / 255.0
    y = to_categorical(y, 3)

    return X, y


def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    X_train, y_train = load_dataset('dataset/train')
    X_test, y_test = load_dataset('dataset/test')

    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    model = build_model()

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=15,
              batch_size=64)

    model.save('models/stress_detector.h5')
    print("Model saved!")
