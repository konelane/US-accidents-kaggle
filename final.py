#! /usr/bin/env python3.6
#import findspark
#findspark.init('/usr/lib/spark-current')

import pyspark
from pyspark.sql import SparkSession
ss = SparkSession.builder.appName("Hehe Final Python Spark with ML").getOrCreate()

#### 1 数据预处理
#### 1.1 读取数据
from pyspark.sql.types import StructField, StructType, StringType,  IntegerType, DoubleType

schema_sdf = StructType([
        StructField('ID', StringType(), True),
        StructField('Source', StringType(), True),
        StructField('TMC', StringType(), True), # 交通消息通道（TMC）代码
        StructField('Severity', StringType(), True), # 事故的严重程度1234 -定序
        StructField('Start_Time', StringType(), True),
        StructField('End_Time', StringType(), True),
        StructField('Start_Lat', DoubleType(), True), # 纬度
        StructField('Start_Lng', DoubleType(), True), # 经度
        StructField('End_Lat', DoubleType(), True),
        StructField('End_Lng', DoubleType(), True),
        StructField('Distance_mi', DoubleType(), True), #受事故影响的道路范围的长度
        StructField('Description', StringType(), True),
        StructField('Number', StringType(), True), # 街道号码
        StructField('Street', StringType(), True), # 街道名称
        StructField('Side', StringType(), True),   # 左侧右侧
        StructField('City', StringType(), True),  # 城市
        StructField('County', StringType(), True), # 县
        StructField('State', StringType(), True),  # 州
        StructField('Zipcode', StringType(), True), # 邮编
        StructField('Country', StringType(), True), # 国家
        StructField('Timezone', StringType(), True), # 时区(eastern,pacific,central...)
        # 站点气象数据-部分数据有
        StructField('Airport_Code', StringType(), True), # 表示一个基于机场的气象站
        StructField('Weather_Timestamp', StringType(), True), # 气象观测时间
        StructField('Temperature_F', DoubleType(), True),
        StructField('Wind_Chill_F', DoubleType(), True),
        StructField('Humidity', DoubleType(), True),
        StructField('Pressure_in', DoubleType(), True),
        StructField('Visibility_mi', DoubleType(), True), # 能见度-英里
        StructField('Wind_Direction', StringType(), True),
        StructField('Wind_Speed_mph', DoubleType(), True),
        StructField('Precipitation_in', DoubleType(), True), # 降水量-如果有
        StructField('Weather_Condition', StringType(), True),
        # 下面都是0-1变量
        StructField('Amenity', StringType(), True), # 便利设施
        StructField('Bump', StringType(), True),    # 减速带
        StructField('Crossing', StringType(), True), # 十字路口
        StructField('Give_Way', StringType(), True), # 存在give_way
        StructField('Junction', StringType(), True), # 路口
        StructField('No_Exit', StringType(), True), # no_exit 
        StructField('Railway', StringType(), True), # 铁路
        StructField('Roundabout', StringType(), True), # 环装交叉路
        StructField('Station', StringType(), True), # 车站
        StructField('Stop', StringType(), True),  # 公交车站
        StructField('Traffic_Calming', StringType(), True), # 限速？？？
        StructField('Traffic_Signal', StringType(), True), # 交通信号灯
        StructField('Turning_Loop', StringType(), True), # 转弯弯道-圆环
        # 下面都是白天晚上
        StructField('Sunrise_Sunset', StringType(), True), # 基于日升日落的日夜
        StructField('Civil_Twilight', StringType(), True), # 基于市民暮光的日夜
        StructField('Nautical_Twilight', StringType(), True), #基于航海暮光的日夜
        StructField('Astronomical_Twilight', StringType(), True)] # 基于天文暮光的日夜
)

df = ss.read.csv('/data/US_Accidents_June20.csv',schema = schema_sdf,sep = ',',header=True)

dat = df.select([
    'Severity', # 事故的严重程度1234 -定序 -因变量-2以下3以上
    
    'Distance_mi', #受事故影响的道路范围的长度
    'Side',   # 左侧右侧-分类变量
    'State', # 州 CA 20%-分类变量
    'Timezone',  # 时区(eastern,pacific,central...)-分类变量
    # 站点气象数据-部分数据有
    'Temperature_F',
    'Wind_Chill_F',
    'Humidity', 
    'Pressure_in', # 当分类变？ -转变成分类变量（以30为限）
    'Visibility_mi',  # 能见度-英里 
    'Wind_Speed_mph', 
    'Precipitation_in', # 降水量-如果有
    # 下面都是0-1变量
    'Amenity',# 便利设施
    'Bump',   # 减速带
    'Crossing', # 十字路口
    'Give_Way',  # 存在give_way
    'Junction',  # 路口
    'No_Exit',  # no_exit 
    'Railway',  # 铁路
    'Roundabout',  # 环装交叉路
    'Station',  # 车站
    'Stop',   # 公交车站
    'Traffic_Calming',  # 限速？？？
    'Traffic_Signal',  # 交通信号灯
    'Turning_Loop',# 转弯弯道-圆环
    # 下面都是白天晚上
    'Sunrise_Sunset',  # 基于日升日落的日夜
    'Civil_Twilight',  # 基于市民暮光的日夜
    'Nautical_Twilight',  #基于航海暮光的日夜
    'Astronomical_Twilight'# 基于天文暮光的日夜
])

#### 1.2 na处理
from pyspark.sql.functions import col, count, isnan, lit, sum

# 用sql查询每列的na情况
def count_not_null(c, nan_as_null=False):
    """Use conversion between boolean and integer
    - False -> 0
    - True ->  1
    """
    pred = col(c).isNotNull() & (~isnan(c) if nan_as_null else lit(True))
    return sum(pred.cast("integer")).alias(c)

dat.agg(*[count_not_null(c) for c in dat.columns]).show()

dat = df.select([
    'Severity', # 事故的严重程度1234 -定序 -因变量-2以下3以上
    
    'Distance_mi', #受事故影响的道路范围的长度
    'Side',   # 左侧右侧-分类变量
    'State', # 州 CA 20%-分类变量
    'Timezone',  # 时区(eastern,pacific,central...)-分类变量
    # 站点气象数据-部分数据有
    'Temperature_F',
    #'Wind_Chill_F', # 缺失太多了
    'Humidity', 
    'Pressure_in', # 当分类变？ -转变成分类变量（以30为限）
    'Visibility_mi',  # 能见度-英里 
    'Wind_Speed_mph', 
    #'Precipitation_in', # 降水量-如果有
    # 下面都是0-1变量
    'Amenity',# 便利设施
    'Bump',   # 减速带
    'Crossing', # 十字路口
    'Give_Way',  # 存在give_way
    'Junction',  # 路口
    'No_Exit',  # no_exit 
    'Railway',  # 铁路
    'Roundabout',  # 环装交叉路
    'Station',  # 车站
    'Stop',   # 公交车站
    'Traffic_Calming',  # 限速？？？
    'Traffic_Signal',  # 交通信号灯
    'Turning_Loop',# 转弯弯道-圆环
    # 下面都是白天晚上
    'Sunrise_Sunset',  # 基于日升日落的日夜
    'Civil_Twilight',  # 基于市民暮光的日夜
    'Nautical_Twilight',  #基于航海暮光的日夜
    'Astronomical_Twilight'# 基于天文暮光的日夜
]).dropna()


#### 1.3 多水平哑变量处理
import pickle
import pandas as pd
import numpy as np
import os
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# 参考老师代码
# 函数1：使用老师给的代码统计哪些类别归入others（计算累计比率）
def dummy_factors_counts(pdf, dummy_columns):
    '''Function to count unique dummy factors for given dummy columns
    pdf: pandas data frame # 输入pd的dataframe
    dummy_columns: list. Numeric or strings are both accepted.
    return: dict same as dummy columns
    '''
    # Check if current argument is numeric or string
    pdf_columns = pdf.columns # Fetch data frame header
    dummy_columns_isint = all(isinstance(item, int) for item in dummy_columns) #isinstance() 判断item是否是int
    #all()用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是，返回 True，否则返回 False
    if dummy_columns_isint:
        dummy_columns_names = [pdf_columns[i] for i in dummy_columns]
    else:
        dummy_columns_names = dummy_columns
    factor_counts = {}
    for i in dummy_columns_names:
        factor_counts[i] = (pdf[i]).value_counts().to_dict()
    #统计每一列里的不同值的个数
    return factor_counts

# 函数2：合并两个字典，并计算同一key的和（两个字典都有子字典） 工具函数
def cumsum_dicts(dict1, dict2):
    '''Merge two dictionaries and accumulate the sum for the same key where each dictionary
    containing sub-dictionaries with elements and counts.
    '''
    # If only one dict is supplied, do nothing.
    if len(dict1) == 0:
        dict_new = dict2
    elif len(dict2) == 0:
        dict_new = dict1
    else:
        dict_new = {}
        for i in dict1.keys():
            dict_new[i] = dict(Counter(dict1[i]) + Counter(dict2[i]))
    return dict_new
#counter是python计数器类，返回元素取值的字典,且按频数降序

# 函数3：返回了包含处理信息的字典
def select_dummy_factors(dummy_dict, keep_top, replace_with, pickle_file):
    '''Merge dummy key with frequency in the given file
    dummy_dict: dummy information in a dictionary format
    keep_top: list 累计比率的峰值（人为设定）
    '''
    dummy_columns_name = list(dummy_dict)#本身词典里就是取值
    # nobs = sum(dummy_dict[dummy_columns_name[1]].values())#没用到
    factor_set = {}  # The full dummy sets——————注意，是空字典，不是集合
    factor_selected = {}  # Used dummy sets
    factor_dropped = {}  # Dropped dummy sets
    factor_selected_names = {}  # Final revised factors
    for i in range(len(dummy_columns_name)):
        column_i = dummy_columns_name[i] #列名
        factor_set[column_i] = list((dummy_dict[column_i]).keys())#第i列变量的全部水平
        factor_counts = list((dummy_dict[column_i]).values())#第i列的值的个数
        factor_cumsum = np.cumsum(factor_counts)#累加
        factor_cumpercent = factor_cumsum / factor_cumsum[-1]#累积比率
        print(factor_cumpercent)
        factor_selected_index = np.where(factor_cumpercent <= keep_top[i])#top这个是给定的
        factor_dropped_index = np.where(factor_cumpercent > keep_top[i]) 
        # 累计比率超过的设定值，则丢进factor_dropped_index
        factor_selected[column_i] = list(
            np.array(factor_set[column_i])[factor_selected_index])#一列有一堆可用取值
        factor_dropped[column_i] = list(
            np.array(factor_set[column_i])[factor_dropped_index])
        # Replace dropped dummies with indicators like `others`
        if len(factor_dropped_index[0]) == 0:
            factor_new = []
        else:
            factor_new = [replace_with] # 如果空值，替换
        factor_new.extend(factor_selected[column_i])#extend列表末尾一次性追加另一个序列中的多个值
        factor_selected_names[column_i] = [column_i + '_' + str(x) for x in factor_new]
    dummy_info = {
        'factor_set': factor_set,
        'factor_selected': factor_selected,
        'factor_dropped': factor_dropped,
        'factor_selected_names': factor_selected_names}
    pickle.dump(dummy_info, open(os.path.expanduser(pickle_file), 'wb')) # 2进制 写入
    print("dummy_info saved in:\t" + pickle_file)
    return dummy_info #返回了一个包含处理信息的字典


# 最后的函数
def select_dummy_factors_from_file(file, header, dummy_columns, keep_top,
                                   replace_with, pickle_file):
    '''Memory constrained algorithm to select dummy factors from a large file
    对大文件使用内存约束算法选择dummy，一个真正的分布式的算法
    要输入文件路径、表头，要变成哑变量的列，保留的比例
    '''
    dummy_dict = {}
    buffer_num = 0
    with open(file) as f:
        while True:
            buffer = f.readlines(
                1024000)  # 返回最大字节1mb
            if len(buffer) == 0:
                break
            else:
                buffer_list = [x.strip().split(",") for x in buffer]
                buffer_num += 1
                if ((buffer_num == 1) and (header is True)):
                    buffer_header = buffer_list[0]
                    buffer_starts = 1
                else:
                    buffer_starts = 0
                buffer_pdf = pd.DataFrame(buffer_list[buffer_starts:])
                if header is True:
                    buffer_pdf.columns = buffer_header
                dummy_dict_new = dummy_factors_counts(buffer_pdf,
                                                      dummy_columns) # 函数1
                dummy_dict = cumsum_dicts(dummy_dict, dummy_dict_new) # 函数2
    dummy_info = select_dummy_factors(dummy_dict, keep_top, replace_with,
                                      pickle_file) # 函数3
    return (dummy_info)


# 看看哪些哑变量需要被替代，设定累积比率，顺便处理空值
file = os.path.expanduser("./small.csv")
header = True
dummy_columns = ['State','Timezone']
keep_top = [0.93,0.93] # 保证最大80%的累计比率
replace_with = 'filter_OTHERS'
pickle_file = os.path.expanduser("~/dummy_info_cp.pkl") # os.path.expanduser可以用~号代替 /home/devel
dummy_info = select_dummy_factors_from_file(file, header, dummy_columns,keep_top, replace_with,pickle_file)

print("哑变量的信息如下：\n",dummy_info)

#### 1.4 整合现有数据
def truetrans(x):
    return(dat[x]=='TRUE')
def daytrans(x):
    return(dat[x]=='Day')


dat_new = dat.select([
    (dat['Severity']>2).alias('Severe_condition'),
    #'Severity', # 事故的严重程度 -因变量 - 严重true
    'Distance_mi',
    (dat['Side']=='R').alias('Side_R'),
    (dat['State']=='CA').alias('Is_CA'), # 根据之前查询结果，此处进行处理
    (dat['Timezone']=='US/Pacific').alias('Is_US_Pacific') ,
    
    # 站点气象数据-定量
    'Temperature_F',
    'Humidity', 
    (dat['Pressure_in']>30).alias('higher_pressure') ,
    'Visibility_mi',  # 能见度-英里 
    'Wind_Speed_mph', 
    
    # 下面都是0-1变量
    truetrans('Amenity').alias('Amenity'),# 便利设施
    truetrans('Bump').alias('Bump'),   # 减速带
    truetrans('Crossing').alias('Crossing'), # 十字路口
    truetrans('Give_Way').alias('Give_Way'),  # 存在give_way
    truetrans('Junction').alias('Junction'),  # 路口
    truetrans('No_Exit').alias('No_Exit'),  # no_exit 
    truetrans('Railway').alias('Railway'),  # 铁路
    truetrans('Roundabout').alias('Roundabout'),  # 环装交叉路
    truetrans('Station').alias('Station'),  # 车站
    truetrans('Stop').alias('Stop'),   # 公交车站
    truetrans('Traffic_Calming').alias('Traffic_Calming'),  # 限速？？？
    truetrans('Traffic_Signal').alias('Traffic_Signal'),  # 交通信号灯
    truetrans('Turning_Loop').alias('Turning_Loop'),# 转弯弯道-圆环
    # 下面都是白天晚上
    daytrans('Sunrise_Sunset').alias('Sunrise_Sunset'),  # 基于日升日落的日夜
    daytrans('Civil_Twilight').alias('Civil_Twilight'),  # 基于市民暮光的日夜
    daytrans('Nautical_Twilight').alias('Nautical_Twilight'),  #基于航海暮光的日夜
    daytrans('Astronomical_Twilight').alias('Astronomical_Twilight')# 基于天文暮光的日夜
])


#### 1.5 转变boolean型变量

# 需要转变的列
true_cols = [    
    'higher_pressure',
    'Side_R',
    'Is_US_Pacific',
    'Is_CA',
    'Amenity',# 便利设施
    'Bump',   # 减速带
    'Crossing', # 十字路口
    'Give_Way',  # 存在give_way
    'Junction',  # 路口
    'No_Exit',  # no_exit 
    'Railway',  # 铁路
    'Roundabout',  # 环装交叉路
    'Station',  # 车站
    'Stop',   # 公交车站
    'Traffic_Calming',  # 限速？？？
    'Traffic_Signal',  # 交通信号灯
    'Turning_Loop',# 转弯弯道-圆环
    # 下面都是白天晚上
    'Sunrise_Sunset',  # 基于日升日落的日夜
    'Civil_Twilight',  # 基于市民暮光的日夜
    'Nautical_Twilight',  #基于航海暮光的日夜
    'Astronomical_Twilight',# 基于天文暮光的日夜
    
    # 因变量
    'Severe_condition'
]

changedTypedf = dat_new.withColumn("Amenity_int", dat_new["Amenity"].cast(IntegerType()))
for a in true_cols:
    changedTypedf = changedTypedf.withColumn(a + "_int", dat_new[a].cast(IntegerType()))
    changedTypedf = changedTypedf.drop(a)
print('\n转变完成的DataFrame：\n',changedTypedf.show())


#### 2 数据透视
# 白天晚上-能见度-严重程度 均值
# groupBy为行
a = changedTypedf.groupBy("Sunrise_Sunset_int").pivot("Severe_condition_int").mean('Visibility_mi').show() # 可以实现透视表
print(a)
# 时区-严重程度-能见度 均值
a = changedTypedf.groupBy("Is_US_Pacific_int").pivot("Severe_condition_int").mean('Visibility_mi').show()
print(a)
# 车道位置-严重程度-能见度 均值
a = changedTypedf.groupBy("Side_R_int").pivot("Severe_condition_int").mean('Visibility_mi').show()
print(a)
# 路左路右-严重程度-昼夜 数量
a = changedTypedf.groupBy("Side_R_int").pivot("Sunrise_Sunset_int").sum('Severe_condition_int').show()
print(a)
# 日出日落 - 严重程度 - 平均风速
a = changedTypedf.groupBy("Sunrise_Sunset_int").pivot("Severe_condition_int").mean('Wind_Speed_mph').show()
print(a)



#### 3 建立模型
#### 3.1 模型参数初始化

random_seed = 250  # 初始化随机种子
training_data_ratio = 0.7   # 训练集与测试集分隔比例

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#### 3.2 建模前数据准备

# 将数据转化为RDD格式
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

transformed_dat = changedTypedf.rdd.map(lambda row:LabeledPoint(row['Severe_condition'],Vectors.dense(row[:-1])))

# 训练集、测试集
splits =[training_data_ratio, 1.0-training_data_ratio]
training_data,test_data = changedTypedf.randomSplit(splits, random_seed) # 对RDD格式的数据进行操作
print("Number of training setrow:%d"%training_data.count())
print("Number oftest set rows:%d"%test_data.count())


#### 3.3 随机森林模型
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline 
from pyspark.ml.regression import RandomForestRegressor

featuresArray = changedTypedf.columns[:-1]
assembler = VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
# 设置maxCategories，使具有>3个不同水平的特征，就被视为连续变量
featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(3)
rf = RandomForestRegressor(maxDepth=4).setLabelCol('Severe_condition_int').setFeaturesCol("indexedFeatures").setNumTrees(10)

pipeline = Pipeline().setStages([assembler,featureIndexer, rf]) 
model = pipeline.fit(training_data) 
# 计算预测值
predictions = model.transform(test_data) 
print('\n***模型的结果展示如下***：\n',predictions.select('prediction', 'Severe_condition_int', 'features').distinct().show() )
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

rfModel = model.stages[1]
print(rfModel)
forestModel = model.stages[2]
print("训练好的随机森林模型:\n" + str(forestModel.toDebugString))

#### 3.4 模型评价

# 使用混淆矩阵评估模型性能[[TP,FN],[TN,FP]]
TP = predictions.filter(predictions['prediction'] >= 0.5).filter(predictions['Severe_condition_int'] == 1).count()
FN = predictions.filter(predictions['prediction'] < 0.5).filter(predictions['Severe_condition_int'] == 1).count()
TN = predictions.filter(predictions['prediction'] < 0.5).filter(predictions['Severe_condition_int'] == 0).count()
FP = predictions.filter(predictions['prediction'] >= 0.5).filter(predictions['Severe_condition_int'] == 0).count()

# 计算准确率 （TP+TN)/(TP+TN+FP+FN)
acc =(TP+TN)/(TP+TN+FP+FN)
# 计算召回率 TP/（TP+TN）
recall = TP/(TP+TN)
print('手动计算准确率 acc[{}]'.format(acc))
print('手动计算召回率 recall[{}]'.format(recall))

# 评估预测的结果
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# AUC为roc曲线下的面积，AUC越接近与1.0说明检测方法的真实性越高
auc = BinaryClassificationEvaluator(rawPredictionCol = 'prediction',labelCol='Severe_condition_int').evaluate(predictions)
print('Spark评估模型准确率 acc[{}], auc[{}]'.format(acc,auc))