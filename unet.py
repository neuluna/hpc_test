from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def convblock(x, filters):
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)

    return x


def UNet(filters=16, layers=4, input_shape=(224, 224, 1), classes=1):
    model_input = Input(input_shape)
    x = model_input

    to_concat = []

    for l in range(layers):
        x = convblock(x, filters*2**l)
        x = convblock(x, filters*2**l)
        to_concat.append(x)
        x = MaxPooling2D()(x)

    x = convblock(x, filters*2**(l+1))

    for l in range(layers-1, -1, -1):
        x = UpSampling2D()(x)
        x = Concatenate()([x, to_concat.pop()])
        x = convblock(x, filters*2**l)
        x = convblock(x, filters*2**l)

    x = Conv2D(classes, 1, padding='same', activation='sigmoid')(x)
    
    return Model(model_input, x)