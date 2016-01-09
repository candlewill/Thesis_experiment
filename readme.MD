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
 |Arousal|0.862|0.757|0.382|0.015|0.342|0.928|


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

 提前停止防止过拟合，early_stopping （论文中解释下）

 Arousal迭代过程：

  ```
 1607/1607 [==============================] - 8s - loss: 1.1588 - val_loss: 0.7473
 Epoch 2/50
 1607/1607 [==============================] - 8s - loss: 0.9726 - val_loss: 0.8876
 Epoch 3/50
 1607/1607 [==============================] - 8s - loss: 0.9028 - val_loss: 0.7789
 Epoch 4/50
 1607/1607 [==============================] - 8s - loss: 0.8521 - val_loss: 0.7159
 Epoch 5/50
 1607/1607 [==============================] - 8s - loss: 0.8311 - val_loss: 0.7107
 Epoch 6/50
 1607/1607 [==============================] - 8s - loss: 0.8015 - val_loss: 0.7416
 Epoch 7/50
 1607/1607 [==============================] - 8s - loss: 0.7632 - val_loss: 0.7503
 Epoch 8/50
 1607/1607 [==============================] - 8s - loss: 0.7756 - val_loss: 0.7773
 Epoch 9/50
 1607/1607 [==============================] - 8s - loss: 0.7465 - val_loss: 0.7585
 Epoch 10/50
 1607/1607 [==============================] - 8s - loss: 0.7176 - val_loss: 0.8485
 Epoch 11/50
 1607/1607 [==============================] - 8s - loss: 0.7184 - val_loss: 0.7312
 Epoch 12/50
 1607/1607 [==============================] - 8s - loss: 0.7012 - val_loss: 0.7279
 Epoch 13/50
 1607/1607 [==============================] - 8s - loss: 0.7026 - val_loss: 0.7475
 Epoch 14/50
 1607/1607 [==============================] - 8s - loss: 0.7066 - val_loss: 0.7630
 Epoch 15/50
 1607/1607 [==============================] - 8s - loss: 0.6801 - val_loss: 0.7305
 Epoch 16/50
 1607/1607 [==============================] - 8s - loss: 0.6680 - val_loss: 0.7572
  ```
 部分预测值和实际值之间的关系

 <p align="center"> <img src="./images/Comparison of predicted and true Arousal.png" height="300" /> </p>

 Arousal迭代过程

 <p align="center"> <img src="./images/arousal迭代过程.png" height="300" /> </p>

3. 使用随机词向量预测CVAT2.0 VA值

 |Metrics|MSE|MAE|pearson_r|R2|Spearman_r|rMSE|
 |----|----|----|----|----|----|----|
 |Valence|1.227|0.896|0.611|0.355|0.636|1.108|
 |Arousal|1.042|0.827|0.292|-0.190|0.250|1.021|

 Valence

 <p align="center"> <img src="./images/valence_rand.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_rand_iter.png" height="300" /> </p>

 Arousal

 <p align="center"> <img src="./images/arousal_rand.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_rand_iter.png" height="300" /> </p>