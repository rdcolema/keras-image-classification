import numpy as np

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16BN():
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""

    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64):
        self.weights_file = 'vgg16_bn.h5'  # download from: http://www.platform.ai/models/
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.build()

    def predict(self, data):
        return self.model.predict(data)

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    def build(self, ft=True):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(self.n_classes, activation='softmax'))

        model.load_weights(self.weights_file)

        if ft:
            self.finetune()

        self.compile()

    def finetune(self):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable=False
        model.add(Dense(self.n_classes, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer=Adam(lr=self.lr),
                loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=None, aug=False):
        if aug:
            train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                               channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                               horizontal_flip=True)
        else:
            train_datagen = ImageDataGenerator()

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                      class_mode='categorical', shuffle=True)

        val_gen = ImageDataGenerator().flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                           class_mode='categorical', shuffle=True)

        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        if aug:
            test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                               channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                               horizontal_flip=True)
        else:
            test_datagen = ImageDataGenerator()

        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)

        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames
