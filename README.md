中文预训练RoBERTa模型 tensorflow/Pytorch版本

中文预训练RoBERTa模型 RoBERTa for Chinese
-------------------------------------------------

24层base版(roberta_l24_zh_base）下载地址：TODO

base版训练数据：10G文本，包含新闻、社区问答、百科数据等。

发布计划：
-------------------------------------------------
1、24层RoBERTa模型(roberta_l24_zh)，使用30G文件训练，        9月8日

2、12层RoBERTa模型(roberta_l12_zh)，使用30G文件训练，        9月8日

3、6层RoBERTa模型(roberta_l6_zh)， 使用30G文件训练，         9月8日

4、PyTorch版本的模型(roberta_l6_zh_pytorch)                9月8日

5、30G中文语料，预训练格式，可直接训练(bert,xlent,gpt2)       9月8日

6、测试集测试和效果对比                                     9月14日

RoBERTa中文版
-------------------------------------------------
本项目所指的中文预训练RoBERTa模型只指按照RoBERTa论文主要精神训练的模型。包括：

1、数据生成方式和任务改进：取消下一个句子预测，并且数据连续从一个文档中获得(见：Model Input Format and Next Sentence Prediction，DOC-SENTENCES)

2、更大更多样性的数据：使用30G中文训练，包含3亿个句子，100亿个字(即token）。由于新闻、社区讨论、多个百科，保罗万象，覆盖数十万个主题，

所以数据具有多样性（为了更有多样性，可以可以加入网络书籍、小说、故事类文学、微博等）。

3、训练更久：总共训练了近20万，总共见过近16亿个训练数据(instance)； 在Cloud TPU v3-256 上训练了24小时，相当于在TPU v3-8(128G显存)上需要训练一个月。

4、更大批次：使用了超大（8k）的批次batch size。

5、调整优化器参数。

除以上外，本项目中文版，使用了全词mask(whole word mask)。在全词Mask中，如果一个完整的词的部分WordPiece子词被mask，

则同属该词的其他部分也会被mask，即全词Mask。

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |



-------------------------------------------------
本项目受到 TensorFlow Research Cloud (TFRC) 资助
