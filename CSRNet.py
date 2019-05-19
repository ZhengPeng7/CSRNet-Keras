from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, concatenate, add, UpSampling2D
from keras.models import Model
from keras.initializers import RandomNormal


def CSRNet(input_shape=(None, None, 3)):

    input_flow = Input(shape=input_shape)
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

    # front-end
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_flow)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # back-end
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)

    output_flow = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    model = Model(inputs=input_flow, outputs=output_flow)

    front_end = VGG16(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 10:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    return model
