import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_deeper(input_shape, learning_rate=0.001, l2_reg=1e-4):
    """
    Улучшенная глубокая CNN:
    - 4 свёрточных блока с L2-регуляризацией
    - BatchNormalisation и Dropout после каждого блока
    - Полносвязный слой 256 нейронов
    """
    inputs = layers.Input(shape=input_shape)

    # Блок 1
    x = layers.Conv1D(64, 7, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Блок 2
    x = layers.Conv1D(128, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Блок 3
    x = layers.Conv1D(256, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Блок 4
    x = layers.Conv1D(512, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.3)(x)

    # Полносвязная часть
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_lstm_deeper(input_shape, learning_rate=0.001, l2_reg=1e-4):
    """
    Трёхслойная LSTM:
    - 256 → 128 → 64 нейрона
    - BatchNormalisation и Dropout после каждого слоя
    - L2-регуляризация на веса и рекуррентные связи
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.LSTM(256, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    recurrent_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    recurrent_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    recurrent_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_bilstm(input_shape, learning_rate=0.001, l2_reg=1e-4):
    """
    Гибрид CNN + двунаправленная LSTM:
    - 3 свёрточных блока (64,128,256)
    - 2 слоя BiLSTM (128,64)
    - L2-регуляризация везде
    """
    inputs = layers.Input(shape=input_shape)

    # Свёрточная часть
    x = layers.Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # BiLSTM часть
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                         kernel_regularizer=regularizers.l2(l2_reg),
                                         recurrent_regularizer=regularizers.l2(l2_reg)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(64,
                                         kernel_regularizer=regularizers.l2(l2_reg),
                                         recurrent_regularizer=regularizers.l2(l2_reg)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Полносвязная часть
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
