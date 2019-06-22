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

# 实验说明：  
1、当训练数据(train_data)的量为100时，由上图可见，方案一、方案二、方案三和方案四在验证集上的预测准确度都差不多，大约在50%~51%左右(随机预测的准确率基准50%)。《python 深度学习》上说，当训练数据在100~200时，方案二的准确率有时可以在58%~60%。即这时方案二可能会比方案一略好。  
2、当训练数据(train_data)的量为1000时，由上图可见，方案一在验证集上的预测准确率65%左右，而方案二、方案三和方案四准确率50%。  
3、当训练数据(train_data)的量为10000时，由上图可见，方案一在验证集上的预测准确率84%左右，而方案二、方案三和方案四准确率50%。  
# 经验总结：
1、当训练集的数量很少时，即直接学习词嵌入和文本情感倾向分类同时进行时，准确率很低，这时可以考虑用Word2Vec来预学习词嵌入，然后将其运用于文本情感倾向分类。这时预测准确率或许会有适当的提高！
2、当训练集数量达到一定量之后，直接学习词嵌入和文本情感倾向分类同时进行的优势将会明显好于用Word2Vec来预学习词嵌入，这时将不再考虑用Word2Vec来预学习词嵌入。
3、方案二、方案三和方案四在不同数量的训练集上表现无明显的区别。表明使用预训练的词嵌入将会明显的限制网络学习文本情感倾向分类任务的效果。  
4、词嵌入的质量与学习任务有关，同样的词在不同学习任务下，词嵌入是明显不同的。比如：这里学习词嵌入与文本情感倾向二元分类任务同时进行，那么学到的词嵌入将明显有二元分类，即：词的positive倾向和negtive倾向将被放大，而词的其他属性将被适当地忽略。具体如下图：  
