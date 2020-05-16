import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop
import cv2

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    '/home/unnamed/Documents/TensorflowProject/rps2/train',
    target_size=(500,500),
    batch_size=20,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    '/home/unnamed/Documents/TensorflowProject/rps2/test',
    target_size=(500,500),
    batch_size=20,
    class_mode='categorical'
)

cdmodel = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(500,500,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

cdmodel.summary()
if(os.path.exists('/home/unnamed/Documents/TensorflowProject/hand-weights.h5')):
    cdmodel.load_weights('hand-weights.h5')
else:
    cdmodel.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = cdmodel.fit_generator(train_generator, epochs=20, validation_data=test_generator, verbose=1)
    cdmodel.save('hand-weights.h5')

# UNCOMMENT BELOW FOR CAMERA USE

cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FPS, 60)

counter = 0
while cap.isOpened():
    counter = counter + 1
    ret, frame = cap.read()
    strFrame = frame
    
    frame = cv2.resize(frame, (500,500))
    frame = np.expand_dims(frame, axis=0)
    frame = np.vstack([frame])
    frame = frame/255
    prediction=cdmodel.predict(frame)
    if(counter == 1):
        if(prediction[0][1] == max(prediction[0])):
            cv2.putText(strFrame, "PAPER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            # print('paper')
        elif(prediction[0][0] == max(prediction[0])):
            cv2.putText(strFrame, "ROCK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            # print('rock')
        elif(prediction[0][2] == max(prediction[0])):
            cv2.putText(strFrame, "SCISSOR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            # print('scissor')
        # elif(prediction[0][3] == max(prediction[0])):
        #     cv2.putText(strFrame, "3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][4] == max(prediction[0])):
        #     cv2.putText(strFrame, "4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][5] == max(prediction[0])):
        #     cv2.putText(strFrame, "5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][6] == max(prediction[0])):
        #     cv2.putText(strFrame, "6", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][7] == max(prediction[0])):
        #     cv2.putText(strFrame, "7", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][8] == max(prediction[0])):
        #     cv2.putText(strFrame, "8", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        # elif(prediction[0][9] == max(prediction[0])):
        #     cv2.putText(strFrame, "9", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        #     # print('scissor')
        counter = 0
    cv2.imshow('Hand-signs', strFrame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
