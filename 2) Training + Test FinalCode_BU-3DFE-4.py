from tensorflow.python.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, Add, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dropout, Input, BatchNormalization, GlobalAveragePooling3D, concatenate
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
# from sklearn.model_selection import train_test_split
from vit_keras import vit
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import os


#vedere la batch size

def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x, downsample, filters, kernel_size):
    y = Conv3D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv3D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv3D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net(width=112, height=112, depth=56):
    inputs = Input(shape=(width, height, depth, 3))
    num_filters = 32

    t = BatchNormalization()(inputs)
    t = Conv3D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters, kernel_size=3)
        num_filters *= 2

    outputs = GlobalAveragePooling3D()(t)
    # t = Flatten()(t)

    model = Model(inputs, outputs)

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


def get_model2D(width=224, height=224):  # ho ridotto dimensione da 224 a 112, ma ho ancora problemi di memoria
    """Build a 2D convolutional neural network model."""

    inputs = Input((width, height, 3))

    x = Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPool2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)

    # outputs = Dense(units=3, activation="softmax")(x)
    # Define the model.
    model = Model(inputs, x, name="2D-CNN")
    return model


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Bosphorus: from MatLab
    all_images = []
    all_labels = []
    all_names = []
    X_train = []
    Y_train = []
    Z_train = []

    X_test = []
    Y_test = []
    Z_test = []

    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']  #, 'Neutral']

    input_3D_train = '..\\Dataset\\Articolo\\BU-3DFE-4\\Training\\'
    input_3D_test = '..\\Dataset\\Articolo\\BU-3DFE-4\\Validation\\'

    input_BU_2D = '..\\Dataset\\BU-3DFE\\2D\\Emotion\\'

    for clas in range(len(classes)):
        x_dict = np.load(input_3D_train + 'X_train_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        X = x_dict['arr_0']
        y_dict = np.load(input_3D_train + 'Y_train_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        Y = y_dict['arr_0']
        z_dict = np.load(input_3D_train + 'Z_train_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        Z = z_dict['arr_0']

        for ii in range(X.shape[0]):
            X_train.append(X[ii])
            Y_train.append(clas)
            Z_train.append(Z[ii])

    for clas in range(len(classes)):
        x_dict = np.load(input_3D_test + 'X_valid_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        X = x_dict['arr_0']
        y_dict = np.load(input_3D_test + 'Y_valid_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        Y = y_dict['arr_0']
        z_dict = np.load(input_3D_test + 'Z_valid_BU-3DFE-4_56_{}.npz'.format(classes[clas]))
        Z = z_dict['arr_0']

        for ii in range(X.shape[0]):
            X_test.append(X[ii])
            Y_test.append(clas)
            Z_test.append(Z[ii])

    X_train3D = np.asarray(X_train)
    Y_train3D = np.asarray(Y_train)
    Z_train3D = np.asarray(Z_train)

    Y_train3D = to_categorical(Y_train3D, len(classes))

    X_val3D = np.asarray(X_test)
    Y_val3D = np.asarray(Y_test)
    Z_val3D = np.asarray(Z_test)

    Y_val3D = to_categorical(Y_val3D, len(classes))

    X_train3D, Y_train3D, Z_train3D = shuffle(X_train3D, Y_train3D, Z_train3D)

    # # divido 80-20 train-valid
    # X_train3D, X_val3D, Y_train3D, Y_val3D, Z_train3D, Z_val3D = train_test_split(all_images, y_train, all_names,
    #                                                                               test_size=0.2, random_state=42,
    #                                                                               shuffle=True)

    X_train2D = []
    X_val2D = []
    Z_train2D = []
    Z_val2D = []

    for i, str in enumerate(Z_train3D):

        # BU
        if str[6:8] == 'AN':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Anger\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'DI':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Disgust\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'FE':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Fear\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'HA':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Happy\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'SA':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Sadness\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'SU':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Surprise\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

        elif str[6:8] == 'NE':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Neutral\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_train2D.append(orig_img)
            Z_train2D.append(new_str)
            continue

    for i, str in enumerate(Z_val3D):

        # BU
        if str[6:8] == 'AN':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Anger\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'DI':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Disgust\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'FE':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Fear\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'HA':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Happy\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'SA':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Sadness\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'SU':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Surprise\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

        elif str[6:8] == 'NE':
            new_str = str.split('xyz_rgb.txt')
            new_str = new_str[0] + 'F2D.png'
            orig_img = cv2.imread(input_BU_2D + 'Neutral\\{}'.format(new_str))
            # orig_img = preprocess_input(orig_img)
            orig_img = cv2.resize(orig_img, (224, 224))
            # orig_img = orig_img / 255.0
            X_val2D.append(orig_img)
            Z_val2D.append(new_str)
            continue

    # for img in X_val2D:
    #     cv2.imshow("a", img)
    #     cv2.waitKey()


    X_train2D = np.asarray(X_train2D)
    #in caso di Vit usare vit.preprocess_inputs()
    X_train2D = preprocess_input(X_train2D)
    Z_train2D = np.asarray(Z_train2D)
    X_val2D = np.asarray(X_val2D)
    X_val2D = preprocess_input(X_val2D)
    Z_val2D = np.asarray(Z_val2D)

    # Visualize an augmented CT scan.
    # data = train_dataset3D.take(1)
    # images, labels = list(data)[0]
    # images = images.numpy()
    # image = images[0]
    # print("Dimension of the CT scan is:", image.shape)

    # plt.imshow(np.squeeze(cv2.cvtColor(image[:, 48, :], cv2.COLOR_BGR2RGB)))

    # Keras
    # Build model.

    train = False
    double = True #2D + 3D
    _2d = False
    _3d = False
    #nome salvataggio rete
    name = 'Final-BU-3DFE-4_3D+2D_b32_pretrain_lr0.0001-bm'

    model3D = get_model3D(width=112, height=112, depth=56)
    model2D = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
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

        model = Model(inputs=[model3D.input, model2D.input], outputs=final)

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

    model.summary()

    # Compile model.
    batch_size = 8  # 8 con b16 e b32
    epochs = 80
    initial_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"],
    )

    # Define callbacks.

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, min_delta=1e-4, min_lr=1e-6, mode='max')

    DirLog = "..\\Results\\Logs\\{}".format(name)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=DirLog, histogram_freq=1)  # update_freq='batch'

    best_model_path = '..\\Results\\Models\\{}-bm.h5'.format(name)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, verbose=1)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=20)

    callbacks_list = [tb_callback, checkpoint_cb, early_stopping_cb, reduce_lr]

    # Train the model, doing validation at the end of each epoch
    if train:
        if double:
            model.fit(x=[X_train3D, X_train2D], y=Y_train3D, validation_data=([X_val3D, X_val2D], Y_val3D), epochs=epochs,
                      batch_size=batch_size, callbacks=callbacks_list, shuffle=False)  # batch_size=8 , shuffle=True
        elif _2d:
            model.fit(x=X_train2D, y=Y_train3D, validation_data=(X_val2D, Y_val3D), epochs=epochs,
                      batch_size=batch_size, callbacks=callbacks_list, shuffle=False)  # batch_size=8 , shuffle=True

        elif _3d:
            model.fit(x=X_train3D, y=Y_train3D, validation_data=(X_val3D, Y_val3D), epochs=epochs,
                      batch_size=batch_size, callbacks=callbacks_list, shuffle=False)  # batch_size=8 , shuffle=True

        # model.fit(x=X_train2D, y=Y_train3D, validation_data=(X_val2D, Y_val3D), epochs=epochs,
        #           batch_size=batch_size, callbacks=callbacks_list, shuffle=True)  # batch_size=8 , shuffle=True
        model.save('..\\Results\\Models\\{}.h5'.format(name))

    else:
        model.load_weights("E:\\Multimodal\\Results\\Models\\Final-BU-3DFE-4_3D+2D_ResNet_avg_lr0.0001-bm.h5")

    if _2d:
        print(model.evaluate(x=X_val2D, y=Y_val3D, batch_size=8))

        prediction2D = model.predict(X_val2D, batch_size=8)
        pred_2D = np.argmax(prediction2D, axis=1)
        Y_val2D = np.argmax(Y_val3D, axis=1)

        print("\nConfusion Matrix 2D")
        print(confusion_matrix(Y_val2D, pred_2D))
        print(classification_report(Y_val2D, pred_2D))

    if _3d:
        print(model.evaluate(x=X_val3D, y=Y_val3D, batch_size=8))

        prediction3D = model.predict(X_val3D, batch_size=8)
        pred_3D = np.argmax(prediction3D, axis=1)
        Y_val3D = np.argmax(Y_val3D, axis=1)

        print("\nConfusion Matrix 3D")
        print(confusion_matrix(Y_val3D, pred_3D))
        print(classification_report(Y_val3D, pred_3D))

    if double:
        print(model.evaluate(x=[X_val3D, X_val2D], y=Y_val3D, batch_size=1))

        prediction3D = model.predict([X_val3D, X_val2D], batch_size=1)
        pred_3D = np.argmax(prediction3D, axis=1)
        Y_val3D = np.argmax(Y_val3D, axis=1)

        print("\nConfusion Matrix Multiple")
        print(confusion_matrix(Y_val3D, pred_3D))
        print(classification_report(Y_val3D, pred_3D))

    for i, str in enumerate(Z_val3D):
        img3D = X_val3D[i]
        img2D = X_val2D[i]

        img3D = np.expand_dims(img3D, axis=0)
        img2D = np.expand_dims(img2D, axis=0)

        print(str, np.argmax(model.predict([img3D, img2D])))



    # print(model.evaluate(x=[X_val3D, X_val2D], y=Y_val3D, batch_size=2))
    # # print(model.evaluate(x=X_val2D, y=Y_val3D, batch_size=8))
    #
    # # prediction2D = model.predict(X_val2D, batch_size=8)
    # # pred_2D = np.argmax(prediction2D, axis=1)
    # # Y_val2D = np.argmax(Y_val3D, axis=1)
    #
    # prediction3D = model.predict([X_val3D, X_val2D], batch_size=2)
    # pred_3D = np.argmax(prediction3D, axis=1)
    # Y_val3D = np.argmax(Y_val3D, axis=1)
    #
    # print("\nConfusion Matrix Multiple")
    # print(confusion_matrix(Y_val3D, pred_3D))
    # print(classification_report(Y_val3D, pred_3D))

