import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


labels_list = []
texts = []
word_index = {}

maxlen = 100                 # 在100个单词后面截断评论
training_samples = 200       # 在200个样本上训练
validation_samples = 10000   # 在10000 个样本上验证
max_words = 10000            # 只考虑前10000个最常见的单词

embeddings_index = {}        # GloVe 单词与向量表示的映射关系
embedding_dim = 100          # 词嵌入维度
"""
处理IMDB原始文本数据
"""
def parse_imdb_data():
    imdb_dir = '/Users/cly/Desktop/Study/data/神经网络训练数据/NLP数据/aclImdb/'
    train_dir = os.path.join(imdb_dir, 'train')
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels_list.append(0)
                else:
                    labels_list.append(1)

"""
对IMDB原始数据进行分词处理
"""
def tokenize_IMDB(maxlen, train_samples, validation_samples, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels_list)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:train_samples]
    y_train = labels[:train_samples]
    x_val   = data[train_samples:train_samples+validation_samples]
    y_val   = labels[train_samples:train_samples+validation_samples]

    return x_train, y_train, x_val, y_val

"""
解析Glove.6B文件
"""
def parse_GloVe():
    glove_dir = "/Users/cly/Desktop/Study/data/神经网络训练数据/NLP数据/glove.6B/"

    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    print("Found %s word vectors." % len(embeddings_index))


"""
计算嵌入矩阵
"""
def calculate_embedding_matrix():
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            # 嵌入索引（embeddings_index）中找不到的词，其嵌入向量全为0
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix



if __name__ == "__main__":
    parse_imdb_data()
    # for i in range(len(texts)):
    #     print("word:", texts[i])
    #     print("label:", labels_list[i])
    x_train, y_train, x_val, y_val = tokenize_IMDB(maxlen, training_samples, validation_samples, max_words)
    # print("x_train:", x_train)
    # print("y_train:", y_train)
    # print("x_val:", x_val)
    # print("y_val:", y_val)
    parse_GloVe()
    for word, vector in embeddings_index.items():
        print("word:", word, "\t vector:",vector)