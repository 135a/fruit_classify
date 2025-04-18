import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
image_location = "D:\AI项目\image\images"
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path):
    # 加载并调整尺寸至600x600（与input_shape一致）
    img = image.load_img(img_path, target_size=(400, 400))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # 与训练时rescale=1./255对齐[1,5](@ref)
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度[3,5](@ref)
    return img_array

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
target_size = (400, 400)
# 训练集和验证集生成器
train_generator = datagen.flow_from_directory(
    directory=image_location,
    target_size=target_size,
    batch_size=1,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
validation_generator = datagen.flow_from_directory(
    directory=image_location,
    target_size=target_size,
    batch_size=1,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)



# 模型定义（修正覆盖问题后）
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(400,400,3), activation='relu', kernel_initializer='random_normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(9, activation='softmax'))

# 编译模型（修正优化器参数名）

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # ✅ 使用 learning_rate 而非 lr
    metrics=['accuracy']
)
es=EarlyStopping(monitor='val_accuracy', patience=50,min_delta=0.0005)



# 训练模型

model.fit(
      train_generator,
        steps_per_epoch = len(train_generator), #total number of batches in one train epoch(train observation/batch size; also called iterations per epoch)
        epochs=200,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks=[es]
)
model.save_weights('fruits-1-0.0005.h5')

'''
model.load_weights("fruits-1-0.0005.h5")
x=model.predict(preprocess_image(r"D:\AI项目\image\images\apple fruit\Image_9.jpg"))
dict=['苹果','香蕉','樱桃','奇异果','葡萄','猕猴桃','芒果','橙子','草莓']
print(np.argmax(x))
print(dict[np.argmax(x)])
'''

