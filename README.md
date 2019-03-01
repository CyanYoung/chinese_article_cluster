## Chinese Article Cluster 2018-10

#### 1.preprocess

prepare() 将按类文件保存的数据汇总，clean() 通过 pos_set 进行词性过滤

#### 2.represent

Dictionary() 建立 word2ind，doc2bow() 得到词频特征、转换为 tfidf 词权特征

#### 3.build

通过 lsi、lda 构建主题聚类模型、保存各类的关键词和权重

#### 4.cluster

predict() 词性过滤、输出文档的主题分布，将 lda 填充为定长序列
