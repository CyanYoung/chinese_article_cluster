## Chinese Article Cluster 2018-10

#### 1.preprocess

prepare() 将按类文件保存的数据汇总，jieba.load_userdict() 导入非切分词

filter() 通过 pos_set 进行词性过滤，打乱后划分训练、测试集

#### 2.represent

Dictionary() 建立 word2ind，doc2bow() 得到文档的词频特征

再转换为 tfidf 词权特征，分别通过 lsi、lda 构建主题模型、保存关键词

#### 3.analyze

predict() 词性过滤、输出文档的主题分布，lsi 为定长、lda 为变长

#### 4.cluster

featurize() 得到主题分布、pad() 填充 lda，通过 KMeans() 构建聚类模型
