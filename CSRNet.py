from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, concatenate
from keras.models import Model
from keras.initializers import RandomNormal


def CSRNet(input_shape=(None, None, 3), branch_choice=1):

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
    if branch_choice == 0:
        dilated_flow_0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
        dilated_flow_0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_0)
        dilated_flow_0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_0)
        dilated_flow_0 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_0)
        dilated_flow_0 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_0)
        dilated_flow_0 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=1, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_0)
    elif branch_choice == 1:
        dilated_flow_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
        dilated_flow_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_1)
        dilated_flow_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_1)
        dilated_flow_1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_1)
        dilated_flow_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_1)
        dilated_flow_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_1)
    elif branch_choice == 2:
        dilated_flow_2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
        dilated_flow_2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_2)
        dilated_flow_2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_2)
        dilated_flow_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_2)
        dilated_flow_2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_2)
        dilated_flow_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_2)
    elif branch_choice == 3:
        dilated_flow_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
        dilated_flow_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_3)
        dilated_flow_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_3)
        dilated_flow_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_3)
        dilated_flow_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_3)
        dilated_flow_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilated_flow_3)

    # branches_concat = concatenate([dilated_flow_0, dilated_flow_1, dilated_flow_2, dilated_flow_3])
    # output_flow = Conv2D(1, (1, 1), (1, 1))(branches_concat)

    output_flow = Conv2D(1, 1, strides=(1, 1), padding='same', dilation_rate=4, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(eval('dilated_flow_'+str(branch_choice)))
    # print("input_flow:", input_flow.shape, "output_flow:", output_flow.shape)
    model = Model(inputs=input_flow, outputs=output_flow)

    # Arange the front-end weights from original vgg16
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
