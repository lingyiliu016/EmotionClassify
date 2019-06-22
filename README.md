# EmotionClassifyExperiment
第一步：下载IMDB数据的原始文本，下载GloVe词嵌入  
首先，打开http://mng.bz/0tIo， 下载原始IMDB数据并压缩.  
其次，打开https://nlp.stanford.edu/projects/glove ,下载2014年英文危机百科的预计算嵌入。这个是一个822M的压缩文件，文件名glove.6B.zip,里面包含400000个单词的100维嵌入向量。  
# 模型一：  
在不使用与训练词嵌入的情况下，学习词嵌入与学习文本情感倾向分类同时进行。
# 模型二：  
使用GloVe词嵌入作为Embedding层，然后冻结该层，训练网络来学习文本情感倾向分类。  
# 模型三：
使用GloVe词嵌入作为Embedding层，然后冻结该层，训练网络epochs次后，解冻Embedding层，然后继续训练网络来学习文本情感倾向分类。  
# 模型四：
使用GloVe词嵌入作为Embedding层初始权重，同时学习词嵌入与学习文本情感倾向分类。  
以下是实验结果：  

![Image text](https://github.com/lingyiliu016/EmotionClassifyExperiment/blob/master/WX20190621-175155%402x.png)
