import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2

def create_model():
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(24, 24, 1)),
        layers.Conv2D(24, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(15)]
    )

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model

def train():
    # stat image count
    data_dir = pathlib.Path('dataset')
    batch_size = 64
    img_width = 24
    img_height = 24

    # load images, generate dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # The classification of the dataset, corresponding to how many image classifications there are in the dataset folder
    class_names = train_ds.class_names
    # Save dataset classifications
    np.save('checkpoint/class_names.npy', class_names)

    # Dataset cache processing
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # Create the model
    model = create_model()
    # Train the model, epochs=10, all datasets are trained 10 times
    model.fit(train_ds, epochs=10, validation_data=val_ds)
    # Save the weights after training
    model.save_weights('checkpoint/my_checkpoint')

def predict(model, imgs, class_name):
    label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
    # Predict the picture, get the predicted value
    predicts = model.predict(imgs)
    results = []
    for predict in predicts:
        index = np.argmax(predict) #search max value
        result = class_name[index] # get char
        results.append(label_dict[int(result)])
    return results

if __name__ == '__main__':
    train()

    model = create_model()
    # 加载前期训练好的权重
    model.load_weights('checkpoint/my_checkpoint')
    # 读出图片分类
    class_name = np.load('checkpoint/class_names.npy')
    print(class_name)
    img1=cv2.imread('imgs/img1.png',0) 
    img2=cv2.imread('imgs/img2.png',0) 
    img3=cv2.imread('imgs/img3.png',0) 
    # # img1=cv2.imread('img1.png',0) 
    # # img2=cv2.imread('img2.png',0) 
    # # # img3=cv2.imread('img3.png',0)
    # # # img4=cv2.imread('img4.png',0)
    # # # img5=cv2.imread('img5.png',0)
    # # # img6=cv2.imread('img6.png',0)
    # # # imgs = np.array([img1,img2,img3,img4,img5,img6])
    imgs = np.array([img1,img2, img3])
    results = predict(model, imgs, class_name)
    print(results)