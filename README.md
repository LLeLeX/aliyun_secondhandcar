# aliyun_secondhandcar
阿里云天池赛-数据挖掘入门-二手车价格预测


# 题目
赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。
[题目链接](https://tianchi.aliyun.com/competition/entrance/231784/information)
# 题目分析
1. 数据特征：数据特征共有31个特征，类别特征10个，数值特征21个，其中15个匿名特征均为数值特征
2. 数据量：训练集包含15万条数据，测试集包含5万条数据
3. 题目类型：属于典型的回归问题
4. 评估指标：平均绝对误差MAE
5. 本题的数据挖掘需要更加注重模型的可解释性
# 数据预处理
## 缺失值
|数据集|字段名|缺失数据量|缺失率|
|--|--|--|--|
|train|model|1|6.67e-6|
|train|bodyType|4506|3.004%|
|train|fuelType|8680|5.787%|
|train|gearbox|5981|3.987%|
|train|notRepairedDamage|24324|16.216%|
|test|bodyType|24324|3.008%|
|test|fuelType|1504|5.848%|
|test|gearbox|48032|3.936%|
|test|notRepairedDamage|8069|16.138%|
思考：

0. 缺失的特征——类别型or连续型
1. 什么时候用众数填补（类别型），什么时候用中位数填补（连续型），什么时候用随机森林填补？（缺失的特征较少，特征中有大量缺失数据时）
2. 为什么能用随机森林填补？
原因是：回归问题是基于一系列已知数据来推测目标数据，即已知数据和目标数据之间存在内在联系，如果将目标数据作为已知数据，我们也可以通过模型来推测缺失数据。
3. 多个特征缺失时，如何用随机森林填补？
从缺失值最少的开始填补
4. 本题众数填充和随机森林填充效果差别不大的原因？
本题的缺失值均为类别型特征，因此不太适合用随机森林回归进行填充

|模型|众数填补 | 随机森林填补|
|--|--|--|
|随机森林 |609.346 |606.129 |
| XGBoost| 573.574| 576.835|
| LightGBM|496.775 |463.968 |

## 异常值处理
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613143040590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)
异常值处理
1. 非法日期00月->07/01，因为目的是要计算车龄，单位是年取整，因此用07月填补对结果影响不大
2. power数据超出范围，用最大值代替，猜想是制定数据范围时不合理
3. 对power数据分桶，为什么要分桶->增加鲁棒性（减少异常数据的影响）和可解释性（对二手车价格进行估值一般是按同一范围的行驶里程和发动机功率会被划分到同一个价格区间中）

## 归一化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613143118381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)
归一化、标准化、正则化的区别
> 归一化是将样本的特征值转换到同一量纲下把数据映射到[0,1]或者[-1, 1]区间内，仅由变量的极值决定，因区间放缩法是归一化的一种。
> 标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，转换为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响。标准化后有正有负
> 正则化是调整模型参数，也称为正则项，防止模型的过拟合
> 它们的相同点在于都能取消由于量纲不同引起的误差；都是一种线性变换，都是对向量X按照比例压缩再进行平移。

# 特征工程

## 特征创造
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613144119683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613144136696.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)
数据分布：
1. 方差角度对数据的理解（方差小->数据基本相同->携带的信息量较小->数据倾斜；方差过大->考虑含义->ID->需要进一步分析挖掘）
2.相关性角度：各个特征与预测值之间的相关性，相关性系数接近于0说明对预测值影响较小
3.偏态分布和正态分布：左偏，右偏->纠偏（log），因为许多模型的假设条件就是正态分布

# 建模
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613144221269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613144243759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MzI4Mg==,size_16,color_FFFFFF,t_70)1. 为什么要用集成学习，集成学习的弱学习器选择，为何集成学习要用弱学习器
弱学习器（比随机猜50%稍好）是为了通过模型之间的差异性提高集成学习模型的泛化能力
2. 集成学习分为bagging（随机森林）和boosting（XGBoost，XGBoost运行效率不高因此选用LightGBM）
3. stacking各层的作用是什么？
第一层提取有效特征，第二层对有效特征进行学习，正是由于第二层学习的不是原始数据，通过这个方法降低模型过拟合
4. 在集成学习的基础上进行Stacking模型融合，为什么要进行Stacking，Stacking第二层为什么要用简单的学习器？
1）在特征提取的过程中已经使用了复杂的学习器，因此对于输出层不需要 2）控制模型复杂度
5. 神经网络与Stacking预测结果加和求平均，一般融合有哪些形式（回归：平均，加权；分类：投票）为什么不用加权平均
1）当个体学习器性能差异较大时使用加权平均 2）权重需要在训练集中学习得到
6.单纯的Stacking效果并没有很好的原因
猜想是因为里面的子模型均为树模型，差异性并不大
7. 集成学习与模型融合的区别
集成学习：弱而不同；模型融合：强而不同。但集成学习并不是只能集成弱学习器，也是可以集成强学习器

