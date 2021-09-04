# 数据流转过程
> * a3m->msa (n * L)
> * ffdb, hhr, atab->xyz_t, t1d, t0d
> * L -> idx_pdb
> * msa -> seq
> * xyz_y, t0d->t2d
> * input:
>> * msa
>> * seq
>> idx_pdb
>> * t1d
>> * t2d
# 每个维度的特征说明
> * t0d
>> * 1 * 3
> * t1d
>> * L * 3
>> * atab里边读取到的三个分值(2，3，4列)
> * t2d
>> * L * L * 10
>> * 距离(1)
>> * omega, theta,phi(6),包括sin, cos
>> * t0d的信息(3维度)
# pair特征的获取
> * Templ_emb 使用了transofrmer的encode过程,包括了位置编码
# 特征处理过程
> * seq->seq1hot
> * msa->msa_emb->msa
> * seq, idx->pair
> * msa, pair, seqonehot, idx->mxa, pair, xyz, lddt
> * xyz 获取方式
>> * 根据atab和pdb里边的坐标获取 
# 预测
> * input
>> * msa
>> * pair
>> * seq1hot
>> * idx_pdb
> * DistanceNetwork
>> * resnet_dist
>> * resnet_omega
>> * resnet_theta
>> * resnet_phi
> * 输出四个概率分布
>> * 37, 37, 37, 19
# refine
> * input
>> * msa, prob(130), seq1hot, idx_pdb
> * model
>> * se3 transformer
> * output
>> * xyz, lddt
# fold
> * input
>> * xyz
> * output
>> * xyz
# final
> * add O Cb
# pdb模板
> * 指的是根据atab读取对应的pdb数据，然后把xyz坐标拿出来做参考
> * 坐标信息被转化为距离和角度信息，这样就参考了同源结构
# msa
> * 对msa进行了embedding处理
> * embedding过程中还用了pos embedding
# transformer的使用
> * 先用resnet计算概率分布，然后放到3dtransformer中使用
> * 疑问: 怎样实现可变长信息的
# 疑问
> * resnet怎么保证图像长度一样的
> * t1d的参考方式以及数据内容
> * t0d的参考方式以及数据内容
# 维度问题
> * resnet 只是用来做特征处理，用来生成对应长度的分布,对长度没要求
# 特征提取和数据预处理
> * 两个是两回事
> * 预处理是把输入改成了 t1d, t2d, msa, onehot等信息
> * 特征提取是用模型把这些再做了处理,IterativeFeatureExtractor
> * IterativeFeatureExtractor
>> * 使用se3transformer输出xyz
>>> * 使用graph transformer生成初始的xyz
>>> * 使用Str2Str做进一步处理，这一步使用的是图以及se3transformer
>>> * 使用finalblock进行最后处理，用的也是Str2Str
>>> * str2str同时使用了msa和pair信息 ，以及onehot
>> * Pair2Pair
>>> * 使用transformer的encode进行处理, pair->PairPair->pair
>>> * msa和pair互相attention，然后再更新到msa和pair中
>>> * 使用iter_block_1和iter_block_2,两个均使用transformer的encoder阶段
>>> * final block使用的也是transformer的方式进行encoder。
>>> * 特征提取阶段使用的是transformer的自己attention过程
>>> * 特征提取的时候同时输出xyz坐标信息
>>>> * 输入信息
>>>>> * seq1hot, idx, msa, pair, 相当于是用的一维和二维作为输入
>>>> * xyz使用的是图神经网络获取,算是一个初步的结果
>>>> * 此处使用的是一个基本的图神经网络
>>>> * Str2Str 使用的是se3transformer
# 特征提取总结
> * 特征提取过程已经生成了初步的xyz，使用的是se3transformer以及图transformer
> * 主要做的是对原始输入的特征进行互相attention的过程
> * 使用se3和图transformer的应该是3dtrack
> * msa和pair进行attention的应该是2dtrack
# Refine_Network
> * 最后使用的。
> * distance预测出来之后， 联合msa， onehot等信息，继续进行处理
> * 生成图， 然后使用se3来做
> * regen_net
>> * 用于生成最初的xyz
>> * 使用的是基本的图transformer
> * refine_net
>> * 使用se3tranformer再次做优化
# 折叠过程
> * TRFold
> * 使用模型预测出来的概率分布信息（使用resnet预测出来的)
> * 使用自定义的入参fold_params
> * 使用adam
# se3用来做xyz的优化
# 疑问：msa和pair的特征attention过程
# 疑问：特征到图坐标的转换过程