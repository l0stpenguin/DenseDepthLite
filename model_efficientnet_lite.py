import sys
# from tensorflow.keras.applications import EfficientNetB0
from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, UpSampling2D
from layers import BilinearUpSampling2D
from loss import depth_loss_function
# import efficientnet.keras as efn 
import keras
from keras_applications.efficientnet import EfficientNetB0
# from keras.appl

def create_model_efficientnet_lite(existing='', is_halffeatures=True):
    if len(existing) == 0:
        print('Loading base model (efficientNet-lite-B0)..')
        
        # Encoder Layers
        base_model = load_model('L0_keras_480.h5')
        base_model_output = base_model.get_layer('re_lu')
        print('Base model loaded.',base_model_output,base_model_output.output.shape)

        # Starting point for decoder
        base_model_output_shape = base_model_output.output.shape #15, 20, 1280

        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1]) / 2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = UpSampling2D(size=(2, 2), name=name + '_upsampling2d')(tensor)
            up_i = Concatenate(name=name + '_concat')(
                [up_i, base_model.get_layer(concat_with).output])  # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                         name='conv2')(base_model_output.output)

        decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='efficientnet-lite0/model/blocks_5/Relu6_1_0') #30, 40, 256
        decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='efficientnet-lite0/model/blocks_3/Relu6_1_0') # 60, 80, 128
        decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='efficientnet-lite0/model/blocks_3/Relu6_0') #120, 160, 64
        decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='efficientnet-lite0/model/blocks_1/Relu6_0')  #240, 320, 64
        if False: decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the model
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')

    return model