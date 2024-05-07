
# LAMOST-spectrum-classification
On the task of classifying LAMOST DR9 spectra, including the three classes of 'GALAXY', 'QSO' and 'STAR'. Very necessary processing was done on the training data, including spectral normalisation, training sample equalisation, etc. Using a suitable integrated 1D CNN as a classifier, which performs well on the validation set without “collapsing”.  



我们更新了训练和验证情况，总共训练了6个epoch，'GALAXY', 'QSO' and 'STAR'的样本量均为3500（STAR每趟训练随机采样），保留10%的数据作为验证。

第6个epoch的混淆矩阵如下：
![epoch6](https://github.com/JovanYe/LAMOST-spectrum-classification/assets/162402413/81d4327e-455a-4316-9e79-57a80503e115)
样本量较少的类星体也获得了出色的准确率


训练过程中损失如下：
![6](https://github.com/JovanYe/LAMOST-spectrum-classification/assets/162402413/dc5924c8-1a6b-43ad-a232-b8a2518f7188)


总正确率以及总召回率如下：
![6](https://github.com/JovanYe/LAMOST-spectrum-classification/assets/162402413/1eb436d9-2c27-47f0-99e3-2c05d050255a)






