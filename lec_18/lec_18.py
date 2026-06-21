"""
Нейронные сети
  Свёрточные (конволюционные нейронные сети) [CNN]
      Используются в задачах компьютерного зрения
  Рекурентные нейросети [RNN]
      Распознавание рукописного текста
  Генеративные созтязательные сети
      Создание художественных и музыкальных произведений
  Многослойный перцептон
      Каждый нейрон связан со всеми нейронами из соседних слоёв
      У каждого нейрона есть смещение, а у каждой связи вес, у каждого внутреннего нейрона есть функция активации,
      например функция выпрямления линейных единиц (ReLu) (if x>0, f(x)=x, f(x)=0), где f(x)=x*m+b (m - вес, b - смещение)
"""

# 1. Загрузка изображения
# 2. Масштабирование
# 3. Нормализация
# 4. Выбор модели
# 5. Загрузка изображения в модель и получение предсказания

#from tensorflow.keras.preprocessing import image
#import numpy as np
#import matplotlib.pyplot as plt
#
#img_path="image.jpg"
#img=image.load_img(img_path, target_size=(224, 224))
#img_array=image.img_to_array(img)
#print(img_array.shape)
#print(img_array[100, 100])
#
#from tensorflow.keras.applications.resnet50 import preprocess_input
#
#img_batch=np.expand_dims(img_array, axis=0)
#img_preprocessed=preprocess_input(img_batch)
#print(img_preprocessed[0, 100, 100])
#
#from tensorflow.keras.applications.resnet50 import ResNet50
#
#model=ResNet50()
#prediction=model.predict(img_preprocessed)
#
#from tensorflow.keras.applications.resnet50 import decode_predictions
#
#print(decode_predictions(prediction))

#TRAIN_DATA_DIR="/mnt/c/Python/repository/lec_18/train500"
#VAL_DATA_DIR="/mnt/c/Python/repository/lec_18/test500"
#TRAIN_SAMPLES=500
#VALIDATION_SAMPLES=500
#NUM_CLASSES=2
#IMG_WIDTH=224
#IMG_HEIGHT=224
#BATCH_SIZE=64        #Сколько изображений модель при обучении принимает одновременно
#
## Аугментация данных - процедура увеличения количества данных путём их искажения
#
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import (
#    Input,
#    Flatten,
#    Dense,
#    Dropout,
#    GlobalAveragePooling2D,
#)
#from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
#from tensorflow.keras.optimizers import Adam
#import math
#
## аугментация и нормализация
#train_datagen=image.ImageDataGenerator(
#    preprocessing_function=preprocess_input,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    zoom_range=0.2,
#)
#
## только нормализация
#val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
#
#train_gen=train_datagen.flow_from_directory(
#    TRAIN_DATA_DIR,
#    target_size=(IMG_WIDTH, IMG_HEIGHT),
#    batch_size=BATCH_SIZE,
#    shuffle=True,
#    seed=1,
#    class_mode="categorical",
#)
#
#val_gen=val_datagen.flow_from_directory(
#    VAL_DATA_DIR,
#    target_size=(IMG_WIDTH, IMG_HEIGHT),
#    batch_size=BATCH_SIZE,
#    shuffle=False,
#    class_mode="categorical",
#)
#
#model=MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#for layer in model.layers[:]:
#    layer.trainable=False
#
#input=Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#custom_model = model(input)
#custom_model = GlobalAveragePooling2D()(custom_model)
#custom_model = Dense(64, activation="relu")(custom_model)
#custom_model = Dropout(0.5)(custom_model)
#prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)
#target_model = Model(inputs=input, outputs=prediction)
#
#target_model.compile(
#    loss="categorical_crossentropy",
#    optimizer=Adam(),
#    metrics=["acc"]
#)
#
#num_steps = math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)
#
#target_model.fit(
#    train_gen,
#    steps_per_epoch=num_steps,
#    epochs=7,
#    validation_data=val_gen,
#    validation_steps=num_steps,
#)
#
#print(val_gen.class_indices)
#target_model.save("out_model.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

img=image.load_img("megasobaka.jpg", target_size=(224, 224))

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

img_preprocessed = preprocess_input(img_batch)
model=load_model("out_model.h5")

prediction=model.predict(img_preprocessed)
print(prediction)