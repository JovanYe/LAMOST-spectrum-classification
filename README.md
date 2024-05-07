# LAMOST-spectrum-classification
On the task of classifying LAMOST DR9 spectra, including the three classes of 'GALAXY', 'QSO' and 'STAR'. Very necessary processing was done on the training data, including spectral normalisation, training sample equalisation, etc. Using a suitable integrated 1D CNN as a classifier, which performs well on the validation set without “collapsing”.  

#备注：
由于模型参数量稍大，现在条件下训练困难，因此对test数据集的结果暂时没有出来，待更新...

我们更新了训练和验证情况，总共训练了6个epoch，'GALAXY', 'QSO' and 'STAR'的样本量均为3500（STAR每趟训练随机采样），保留10%的数据作为验证
