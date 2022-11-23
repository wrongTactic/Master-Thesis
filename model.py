from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, Add, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Input, GlobalAveragePooling3D, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def build_model(double, _2d, _3d, verbose, classes):
    #double 2D + 3D if true.

    _2d = False
    _3d = False
    # nome salvataggio rete
    name = 'Final-BU-3DFE-4_3D+2D_b32_pretrain_lr0.0001-bm'

    model3D = get_model3D(width=112, height=112, depth=56)
    model2D = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model3D._name = "model3D"
    model2D._name = "model2D"
    for l in model3D.layers:
        l._name = l._name + model3D._name
    for l in model2D.layers:
        l._name = l._name + model2D._name
    # model2D = vit.vit_b32(
    #     image_size=224,
    #     activation='sigmoid',
    #     pretrained=True,
    #     include_top=False,
    #     pretrained_top=False
    # )
    # model2D = vit.vit_b16(
    #     image_size=224,
    #     # activation='sigmoid',
    #     pretrained=True,
    #     include_top=False,
    #     pretrained_top=False
    # )
    # GA = Dense(units=256, activation="relu")(model2D.output)
    # model2D = Model(inputs=model2D.input, outputs=GA)

    if double:
        final = concatenate([model3D.output, model2D.output])
        # final = Dense(units=512, activation="relu")(final)
        final = Dropout(0.5)(final)
        final = BatchNormalization()(final)
        final = Dense(units=len(classes), activation="softmax")(final)
        inputs_models = [model3D.input]
        inputs_models.append(model2D.input)
        model = Model(inputs=inputs_models, outputs=final)

    elif _2d:
        final = Dropout(0.5)(model2D.output)
        final = BatchNormalization()(final)
        final = Dense(units=len(classes), activation="softmax")(final)
        model = Model(inputs=[model2D.input], outputs=final)

    elif _3d:
        final = Dropout(0.5)(model3D.output)
        final = BatchNormalization()(final)
        final = Dense(units=len(classes), activation="softmax")(final)
        model = Model(inputs=[model3D.input], outputs=final)

    if verbose: model.summary()
    return model

def get_model3D(width=112, height=112, depth=56):
    """Build a 3D convolutional neural network model."""

    inputs = Input((width, height, depth, 3))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)

    # x = Dense(units=512, activation="relu")(x)
    # x = Dropout(0.3)(x)
    # x = Dense(units=512, activation="relu")(x)
    # x = Dropout(0.3)(x)

    # outputs = Dense(units=3, activation="softmax")(x)
    # Define the model.
    model = Model(inputs, x, name="3D-CNN")
    return model
