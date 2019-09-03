RoBERTa for Chinese

中文预训练RoBERTa模型 
-------------------------------------------------

<a href='https://storage.googleapis.com/roberta_zh/roberta_model/roeberta_zh_L-24_H-768_A-12.zip'>24层base版(roberta_l24_zh_base）下载</a>

base版训练数据：10G文本，包含新闻、社区问答、百科数据等。

发布计划 Release Plan：
-------------------------------------------------
1、24层RoBERTa模型(roberta_l24_zh)，使用30G文件训练，        9月8日

2、12层RoBERTa模型(roberta_l12_zh)，使用30G文件训练，        9月8日

3、6层RoBERTa模型(roberta_l6_zh)， 使用30G文件训练，         9月8日

4、PyTorch版本的模型(roberta_l6_zh_pytorch)                 9月8日

5、30G中文语料，预训练格式，可直接训练(bert,xlent,gpt2)        9月8日

6、测试集测试和效果对比                                      9月14日

RoBERTa中文版 Chinese Version
-------------------------------------------------
本项目所指的中文预训练RoBERTa模型只指按照RoBERTa论文主要精神训练的模型。包括：

1、数据生成方式和任务改进：取消下一个句子预测，并且数据连续从一个文档中获得(见：Model Input Format and Next Sentence Prediction，DOC-SENTENCES)

2、更大更多样性的数据：使用30G中文训练，包含3亿个句子，100亿个字(即token）。由于新闻、社区讨论、多个百科，保罗万象，覆盖数十万个主题，

所以数据具有多样性（为了更有多样性，可以可以加入网络书籍、小说、故事类文学、微博等）。

3、训练更久：总共训练了近20万，总共见过近16亿个训练数据(instance)； 在Cloud TPU v3-256 上训练了24小时，相当于在TPU v3-8(128G显存)上需要训练一个月。

4、更大批次：使用了超大（8k）的批次batch size。

5、调整优化器参数。

除以上外，本项目中文版，使用了全词mask(whole word mask)。在全词Mask中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即全词Mask。

dynamic mask在本项目中没有实现

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |

效果测试与对比 Performance 
-------------------------------------------------

### 自然语言推断：XNLI

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) | 
| ERNIE | **79.7 (79.4)** | 78.6 (78.2) | 
| **BERT-wwm** | 79.0 (78.4) | 78.2 (78.0) | 
| **BERT-wwm-ext** | 79.4 (78.6) | **78.7 (78.3)** |
| **RoBERTa** | ? | ? |

###  Sentence Pair Matching (SPM): LCQMC

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | ? | ? | 
| ERNIE | ? | ? | 
| **BERT-wwm** |? | ? | 
| **BERT-wwm-ext** | ? |?  |
| **RoBERTa** | ? | ? |

? 处地方，将会很快更新到具体的值


-------------------------------------------------
本项目受到 TensorFlow Research Cloud (TFRC) 资助 / Project supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)
 
Reference
-------------------------------------------------
1、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>

2、<a href="https://arxiv.org/pdf/1906.08101.pdf">Pre-Training with Whole Word Masking for Chinese BERT</a>