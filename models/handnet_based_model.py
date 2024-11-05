import tensorflow as tf
from tensorflow.keras import layers, models


__all__ = ['handnet_based_model']


def _rf_signal_embedding(x):
    """
    RF-signal Embedding
    """
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# dropoutの追加、フィルタを減らしてキャパシティを落とす

# Block 1: 4つの経路
def _block1(x):
    """
    Block1
    以下の4つの経路を結合して返す
        - path1: Global Spatial Pathway
        - path2: Global Time-Frequency Pathway
        - path3: Local Time-Frequency Pathway
        - path4: Local Spatial Pathway
    """
    # 経路1: Average Pooling → 1x1フィルタ
    path1 = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(path1)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.Activation('relu')(path1)
    path1 = layers.UpSampling2D(size=(x.shape[1], x.shape[2]))(path1)  

    # 経路2: 1x1フィルタ→3x3フィルタ→5x5フィルタ
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)

    # 経路3: 1x1フィルタ→3x3フィルタ
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Activation('relu')(path3)
    path3 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path3)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Activation('relu')(path3)

    # 経路4: 1x1フィルタ
    path4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation('relu')(path4)

    # 4つの経路の出力を結合
    return layers.Concatenate()([path1, path2, path3, path4])


# Block 2: 4つの経路
def _block2(x):
    """
    Block2
    以下の4つの経路を結合して返す
        - path1: Global Spatial Pathway
        - path2: Global Time-Frequency Pathway
        - path3: Local Time-Frequency Pathway
        - path4: Local Spatial Pathway
    """
    # 経路1: Average Pooling → 1x1フィルタ
    path1 = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(path1)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.Activation('relu')(path1)
    path1 = layers.UpSampling2D(size=(x.shape[1], x.shape[2]))(path1)  

    # 経路2: 1x1フィルタ→1x7フィルタ→7x1フィルタ→1x7フィルタ→7x1フィルタ
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)

    # 経路3: 1x1フィルタ→1x7フィルタ→7x1フィルタ
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Activation('relu')(path3)
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path3)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Activation('relu')(path3)
    path3 = layers.Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path3)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Activation('relu')(path3)

    # 経路4: 1x1フィルタ
    path4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation('relu')(path4)

    # 4つの経路の出力を結合
    return layers.Concatenate()([path1, path2, path3, path4])


def _multi_scale_shared_encoder(x, num_block1, num_block2):
    # Block 1を複数回適用
    for _ in range(num_block1):
        x = _block1(x)

    # Block 2を複数回適用
    for _ in range(num_block2):
        x = _block2(x)

    # 深層表現rの生成
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    r = layers.BatchNormalization()(x)

    return r


def _residual_block(x, filters=64, kernel_size=3):
    """
    Residual Block
    """
    residual = x
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # フィルタ数を合わせるために、residualにもConv2Dを適用
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, kernel_size=1, padding='same')(residual)
        residual = layers.BatchNormalization()(residual)
    
    x = layers.add([x, residual])
    x = layers.LeakyReLU(alpha=0.3)(x)  # Using Leaky ReLU
    return x


def _mask_decoder(x, num_residual_blocks):
    """
    Mask Decoder
    """
    x = layers.Dense(units=7*7*1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape(target_shape=(7, 7, 1))(x)

    # Residual Blocksを複数回適用
    for _ in range(num_residual_blocks):
        x = _residual_block(x)
    
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    
    x = layers.Reshape((28, 28))(x)
    return x


def handnet_based_model(input_shape=(52, 10, 2), num_block1=3, num_block2=3, num_residual_blocks=14):
    """
    HandNet-based Model
    """
    # モデルの入力 (F, T, 2) 次元のテンソル
    input_tensor = tf.keras.Input(shape=input_shape)

    # RF Signal Embedding
    x = _rf_signal_embedding(input_tensor)
    
    # Multi-Scale Shared Encoder
    x = _multi_scale_shared_encoder(x, num_block1, num_block2)
    
    # Mask Decoder
    x = _mask_decoder(x, num_residual_blocks)
    
    # モデルの作成
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    
    return model