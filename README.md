## Chinese Article Cluster 2018-10

#### 1.preprocess

prepare() 将按类文件保存的数据汇总，jieba.load_userdict() 导入非切分词

filter() 通过 pos_set 进行词性过滤，打乱后划分训练、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化

#### 3.represent

Dictionary() 建立 word2ind，doc2bow() 得到文档的词频特征

再转换为 tfidf 词权特征，分别通过 lsi、lda 构建主题模型、保存关键词

#### 4.analyze

predict() 词性过滤、输出文档的主题分布，lsi 为定长、lda 为变长

#### 5.build

featurize() 得到主题分布、pad() 填充 lda，通过 KMeans() 构建聚类模型

#### 6.eval

计算各主题中最多类别的比例、求平均作为整体的准确率