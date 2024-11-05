from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Reshape, Conv2DTranspose, Activation, Reshape


# noiseを加えてデータのaugmentation

def csi2image_model(input_shape, latent_dim):
    model = Sequential()
    
    # CSI2Feature layers
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(96, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(latent_dim, activation='linear', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    # Decoder layers
    model.add(Dense(units=7*7*1, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(7, 7, 1)))
    model.add(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(Activation('sigmoid'))
    model.add(Reshape((28, 28)))

    return model