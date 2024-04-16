from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator    

train_dir = 'FishImgDataset/train'
test_dir = 'FishImgDataset/test'
val_dir = 'FishImgDataset/val'

img_width, img_heiht = 400, 400
input_shape = (img_width, img_heiht, 3)

epochs = 30
batch_size = 20
train_samples = 8791
test_samples = 2751
val_samples = 1760

model = Sequential([
    Conv2D(32, (3,3), padding='same',activation='relu', input_shape = input_shape),
    MaxPooling2D((2,2),strides=2),

    Conv2D(64, (3,3), padding='same',activation='relu'),
    MaxPooling2D((2,2),strides=2),

    Flatten(),

    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(31,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width,img_heiht),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width,img_heiht),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width,img_heiht),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps = val_samples // batch_size)

scores = model.evaluate_generator(test_generator, test_samples // batch_size)
print('точность на тестовой выборке' + str(scores[1]))