### 论文实验补充

本项目是硕士毕业论文中部分补充实验

### 主要内容

1. CVAT 2.0 数据统计

2. CVAT 使用word2vec词向量，CNN预测结果

3. CVAT 使用随机词向量，CNN预测结果

4. CVAT 使用不同维度词向量预测结果

### 实验结果

1. CVAT 2.0 数据统计

 |语料库大小|文字总数|平均句子长度|词汇量|标记维度|句子最大长度|句子最小长度|
 |-----|-----|-----|-----|-----|-----|-----|
 |2,009|111,558|55.52|14,708|V+A|247|4|
 
 VA 分布图
 
 <p align="center"> <img src="./images/cvat_va_scatter(small).png" height="350" /> </p>
 
2. 使用CNN、word2vec预测va值

 Hyper-parameters:

 ```python
 N_fm = 100, kernel_size = 3, maxlen = 100, hidden_dims = 100, dropout_p = 0.5, pool_length = pool_length=math.floor((maxlen-kernel_size+1)/2), batch_size = 8, nb_epoch=5
 ```
 Experiement result:

 |Metrics|MSE|MAE|pearson_r|R2|Spearman_r|rMSE|
 |----|----|----|----|----|----|----|
 |Valence|0.773|0.716|0.387|0.117|0.359|0.879|


 部分预测Valence预测图

 <p align="center"> <img src="./images/Comparison of predicted and true Valence.png" height="400" /> </p>

 迭代过程

 |epoch number|loss|val_loss|
 |----|----|----|
 |1|1.1743|0.7978|
 |2|0.9886|0.7697|
 |3|0.8962|0.7139|
 |4|0.8822|0.7076|
 |5|0.8274|0.7158|

 迭代过程图

 <p align="center"> <img src="./images/valence_iter.png" height="300" /> </p>



 