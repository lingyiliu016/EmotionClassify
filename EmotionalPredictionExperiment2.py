from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import keras
import matplotlib.pyplot as plt

import DataPretreat

if __name__ == "__main__":
    """
    解析IMDB文件数据
    """
    DataPretreat.parse_imdb_data()
    """
    获取训练集和验证集
    """
    x_train, y_train, x_val, y_val = DataPretreat.tokenize_IMDB(DataPretreat.maxlen,
                                                                DataPretreat.training_samples,
                                                                DataPretreat.validation_samples,
                                                                DataPretreat.max_words)
    """
    解析Glove.6B文件
    """
    DataPretreat.parse_GloVe()

    """
    计算嵌入矩阵
    """
    embedding_matrix = DataPretreat.calculate_embedding_matrix()

    """
    神经网络结构设计
    """
    model = Sequential()
    model.add(Embedding(DataPretreat.max_words, DataPretreat.embedding_dim, input_length=DataPretreat.maxlen, name='embed'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    """
    将与训练的词嵌入加入到Embedding层中,并冻结Embedding层，使其在训练的过程中不被更新
    """
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    """
    训练与评估
    """

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    callbacks = [keras.callbacks.TensorBoard(log_dir='./log',
                                             histogram_freq=1,
                                             batch_size=32,
                                             embeddings_freq=1,
                                             embeddings_layer_names='embed',
                                             embeddings_data=x_train[:2000].astype('float32'),
                                             update_freq='epoch')]
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)
    model.save('pre_trained_glove_model.h5')

    """
    绘制loss and accurate
    """
    acc      = history.history['acc']
    val_acc  = history.history['val_acc']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


