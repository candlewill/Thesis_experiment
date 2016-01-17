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

 <p align="center"> <img src="./images/cvat_va_scatter(colorful).png" height="350" /> </p>
 
2. 使用CNN、word2vec预测va值

 Hyper-parameters:

 ```python
 N_fm = 100, kernel_size = 5, maxlen = 100, hidden_dims = 100, dropout_p = 0.5, pool_length = pool_length=math.floor((maxlen-kernel_size+1)/2), batch_size = 8, max_nb_epoch=50
 ```
 Experiement result:

 |Metrics|MSE|MAE|pearson_r|R2|Spearman_r|rMSE|
 |----|----|----|----|----|----|----|
 |Valence|1.189|0.874|0.639|0.374|0.662|1.09|
 |Arousal|0.773|0.716|0.387|0.117|0.359|0.879|

 **Valence**

 部分预测Valence预测图

 <p align="center"> <img src="./images/Comparison of predicted and true Valence.png" height="400" /> </p>

 valence迭代过程

 <p align="center"> <img src="./images/valence_iter.png" height="300" /> </p>

 提前停止防止过拟合，early_stopping （论文中解释下）

 valence直方图，绝对误差和频率

 <p align="center"> <img src="./images/hist_valence.png" height="300" /> </p>

 **Arousal**

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

4. 随机产生词向量，不同维度对实验(Valence)的影响

 **10维**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.25836214085 0.910268113036 (0.60011948439809737, 1.0890142012958216e-40) 0.338585582905 SpearmanrResult(correlation=0.63293688023783556, pvalue=2.1862829634462172e-46) 1.1217674183395847
 ```

 <p align="center"> <img src="./images/dim10_valence_comparison.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim10_valence_hist.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim10_valence_iter.png" height="300" /> </p>

 **50维**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.3131630047 0.931548146482 (0.59641990871786488, 4.3503589427249966e-40) 0.309781409411 SpearmanrResult(correlation=0.63282173346108705, pvalue=2.2956331299069574e-46) 1.1459332461809077
 ```

 <p align="center"> <img src="./images/dim50_valence_comparison.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim50_valence_hist.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim50_valence_iter.png" height="300" /> </p>

 **100维**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.30339259081 0.917410494447 (0.59833307930982149, 2.1303463971132947e-40) 0.314916888616 SpearmanrResult(correlation=0.61738254682301019, pvalue=1.3300052107696708e-43) 1.1416622052139032
 ```

 <p align="center"> <img src="./images/dim100_valence_comparison.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim100_valence_hist.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim100_valence_iter.png" height="300" /> </p>

 **200维**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.23475371152 0.882428248917 (0.62016164255507433, 4.3436508709342757e-44) 0.350994535 SpearmanrResult(correlation=0.64549360851831772, pvalue=9.4037467810345631e-49) 1.111194722592499
 ```

 <p align="center"> <img src="./images/dim200_valence_comparison.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim200_valence_hist.png" height="300" /> </p>
 <p align="center"> <img src="./images/dim200_valence_iter.png" height="300" /> </p>

5. RMV方法实验结果

 **Valence**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.2835520318 0.94156873862 (0.57191450528400101, 2.7101874320614917e-36) 0.325345390358 SpearmanrResult(correlation=0.62216714491239822, pvalue=1.9235418374036149e-44) 1.1329395534605917
 ```
 <p align="center"> <img src="./images/valence_rmv.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_rmv_dist.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_rmv_iter.png" height="300" /> </p>

 **Arousal**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 0.853916672335 0.752925855315 (0.275446058880638, 1.971860439753554e-08) 0.0241592254039 SpearmanrResult(correlation=0.25269012595517543, pvalue=2.8334554626819371e-07) 0.9240761182578903
 ```
 <p align="center"> <img src="./images/arousal_rmv.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_rmv_dist.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_rmv_iter.png" height="300" /> </p>

6. RMV-rand实验结果

 **Valence**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.2889382787 0.938735591411 (0.57212649720456332, 2.5207511400979108e-36) 0.322514296476 SpearmanrResult(correlation=0.62359280247652293, pvalue=1.074145807226024e-44) 1.1353141762099406
 ```
 <p align="center"> <img src="./images/valence_rmv_rand.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_rmv_rand_dist.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_rmv_rand_iter.png" height="300" /> </p>

 **Arousal**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 0.862190900749 0.756158571479 (0.27546894983520076, 1.9663319818446003e-08) 0.0147035844416 SpearmanrResult(correlation=0.2557304787319743, pvalue=2.0133710378673852e-07) 0.9285423526954456
 ```
 <p align="center"> <img src="./images/arousal_rmv_rand.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_rmv_rand_dist.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_rmv_rand_iter.png" height="300" /> </p>
 
7. 使用GloVe词向量

 ##### CNN方法

 **Valence**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.1164321125 0.867102129429 (0.68690511510825092, 2.0695044715449238e-57) 0.413186179919 SpearmanrResult(correlation=0.70279031194106056, pvalue=3.9585791260288134e-61) 1.056613511412774
 ```
 <p align="center"> <img src="./images/valence_glove.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_glove_iter.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_glove_hist.png" height="300" /> </p>

 **Arousal**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.03420569678 0.825181955229 (0.32846545402454846, 1.4433020927882028e-11) -0.181871862839 SpearmanrResult(correlation=0.29679482897748571, pvalue=1.2832242048569735e-09) 1.0169590438083371
 ```
 <p align="center"> <img src="./images/arousal_glove.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_glove_iter.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_glove_hist.png" height="300" /> </p>

 ##### RMV

 **Valence**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 1.23910091076 0.924262162726 (0.59081545611658148, 3.4276731557030528e-39) 0.348709580488 SpearmanrResult(correlation=0.64842236586078461, pvalue=2.5424448943081574e-49) 1.113149096373635
 ```
 <p align="center"> <img src="./images/valence_glove_rmv.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_glove_rmv_iter.png" height="300" /> </p>
 <p align="center"> <img src="./images/valence_glove_rmv_hist.png" height="300" /> </p>

 **Arousal**

 ```
 prediction result:
 MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
 0.807147009032 0.733752654548 (0.3155619729987712, 9.5800508818236855e-11) 0.0776067641902 SpearmanrResult(correlation=0.28526844931242679, pvalue=5.771686920239029e-09) 0.8984136068826754
 ```
 <p align="center"> <img src="./images/arousal_glove_rmv.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_glove_rmv_iter.png" height="300" /> </p>
 <p align="center"> <img src="./images/arousal_glove_rmv_hist.png" height="300" /> </p>

8. Additional Experiment

 The above experiment use sigma=1 to exclude the outlier. sigma = 1 is the best way? Maybe we can use sigma = 1.5 or 2.0 as the outlier selection mechanism.

 * sigma=1.5

  **Valence-word2vec**

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  1.17501734133 0.857153185261 (0.64596081596608124, 7.6401374203288254e-49) 0.389370319799 SpearmanrResult(correlation=0.678238647820257, pvalue=1.7600146928857212e-55) 1.0839821683646538
  ```

  **Arousal-word2vec**

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  0.921457808986 0.767747403198 (0.34734165685328955, 7.6746875625596964e-13) -0.115481114959 SpearmanrResult(correlation=0.31769903648833775, pvalue=7.0456692082057659e-11) 0.9599259393233046
  ```

 * sigma=2.0

  **Valence-word2ve**

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  1.23509052786 0.88814520616 (0.62644756090465481, 3.3145568313467529e-45) 0.358578117496 SpearmanrResult(correlation=0.65623542877526542, pvalue=7.2243601114955732e-51) 1.1113462682094133
  ```

  **Arosual-word2ve**

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  0.816087390281 0.732002397982 (0.33742783810355886, 3.6739148370540297e-12) 0.0124040738807 SpearmanrResult(correlation=0.31302769736912855, pvalue=1.3748173925512652e-10) 0.9033755532896222
  ```

  |sigma|MSE|MAE|r|
  |-----|-----|-----|-----|
  |Valence(sigma=1.0)|1.189|0.874|0.639|
  |Valence(sigma=1.5)|1.175|0.857|0.646|
  |Valence(sigma=2.0)|1.235|0.888|0.626|
  |Arousal(sigma=1.0)|0.773|0.716|0.387|
  |Arousal(sigma=1.5)|0.921|0.768|0.347|
  |Arousal(sigma=2.0)|0.816|0.732|0.337|


  RMV method:

  sigma = 1.5
  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  1.25285167125 0.925782175414 (0.59217695709096529, 2.0836660810940871e-39) 0.348921595928 SpearmanrResult(correlation=0.64716559558699882, pvalue=4.4645167616700477e-49) 1.119308568379042
  ```

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  0.811448281476 0.725746071187 (0.27298738031183734, 2.6618926487311099e-08) 0.0176921559236 SpearmanrResult(correlation=0.2505769297144147, pvalue=3.5837527768889621e-07) 0.9008042414840945
  ```

  sigma = 2.0

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  1.25824152441 0.925569502635 (0.58972330837386944, 5.1011083732185485e-39) 0.346555066994 SpearmanrResult(correlation=0.64854313773014549, pvalue=2.4082009431610321e-49) 1.1217136552679012
  ```

  ```
  prediction result:
  MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
  0.80917985945 0.723266230294 (0.27168756180052012, 3.115749106640244e-08) 0.0207632880898 SpearmanrResult(correlation=0.24893271210102805, pvalue=4.2962166603105685e-07) 0.8995442509681696
  ```

  |sigma|MSE|MAE|r|
  |-----|-----|-----|-----|
  |Valence(sigma=1.0)|1.284|0.942|0.572|
  |Valence(sigma=1.5)|1.253|0.926|**0.592**|
  |Valence(sigma=2.0)|1.258|0.926|0.589|
  |Arousal(sigma=1.0)|0.854|0.753|**0.275**|
  |Arousal(sigma=1.5)|0.811|0.726|0.273|
  |Arousal(sigma=2.0)|0.809|0.723|0.272|

9. Worst result

 Let all the predicted valence and arousal be 5, and the performance is:

  |sigma|MSE|MAE|r|
  |-----|-----|-----|-----|
  |Valence(sigma=1.0)|1.945|1.192|nan|
  |Valence(sigma=1.5)|1.971|0.204|nan|
  |Valence(sigma=2.0)|1.968|1.204|nan|
  |Arousal(sigma=1.0)|0.898|0.776|nan|
  |Arousal(sigma=1.5)|0.830|0.731|nan|
  |Arousal(sigma=2.0)|0.831|0.733|nan|

 <p align="center"> <img src="./images/worst_result.png" height="300" /> </p>
 <p align="center"> <img src="./images/worst_result_hist.png" height="300" /> </p>

10. Tradictional Method

  sigma = 1.0 using CVAW lexicon
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||
  
  sigma = 1.5 using CVAW lexicon
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||
  
  sigma = 2.0 using CVAW lexicon
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||
  
  **Extended Lexicon**

  sigma = 1.0 using extended CVAW lexicon (Neural_cand)
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||

  sigma = 1.5 using CVAW lexicon
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||
  
  sigma = 2.0 using CVAW lexicon
  
  |Dims|Mean Method|Regression|MSE|MAE|r|
  |-------|-------|-------|-------|-------|-------|
  |Valence|Geometric|False||||
  |Valence|Arithmetic|False||||
  |Valence|Geometric|True||||
  |Valence|Arithmetic|True||||
  |Arousal|Geometric|False||||
  |Arousal|Arithmetic|False||||
  |Arousal|Geometric|True||||
  |Arousal|Arithmetic|True||||




### Contact me

* [Yunchao He] (https://plus.google.com/+YunchaoHe)
* yunchaohe@gmail.com
* [YZU](http://www.yzu.edu.tw/) at Taiwan
* [Weibo](http://weibo.com/heyunchao)
* [Facebook](https://www.facebook.com/yunchao.h)
* [Twitter](https://twitter.com/candlewill)
* @元智大学资讯工程学系1608B 民105年1月
 