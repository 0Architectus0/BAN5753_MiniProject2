import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc, desc, when
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix
from pyspark.ml.classification import LogisticRegression
%matplotlib inline

spark=SparkSession.builder \
.master ("local[*]")\
.appName("MiniProject2")\
.getOrCreate()

sc=spark.sparkContext
sqlContext=SQLContext(sc)
#####################
## load
#####################
df=spark.read \
 .option("header","True")\
 .option("inferSchema","True")\
 .option("sep",";")\
 .csv("../data/XYZ_Bank_Deposit_Data_Classification.csv")
#####################
## transform
#####################
df_renamed = df.withColumnRenamed("cons.price.idx","cons_price_idx")\
.withColumnRenamed("emp.var.rate", "emp_var_rate")\
.withColumnRenamed("cons.conf.idx","cons_conf_idx")\
.withColumnRenamed("nr.employed", "nr_employed")

df3 = df_renamed.withColumn("wasPreviouslyContacted",when(df_renamed.pdays==999,0).otherwise(1))

df3 = df3.withColumn("pdays",when(df3.pdays==999,np.nan).otherwise(df3.pdays))

df4 = df3.withColumn("marital",when(df3.marital=="unknown",np.nan).otherwise(df3.marital))\
         .withColumn("marital_na_ind",when(df3.marital=="unknown",1).otherwise(0))\
         .withColumn("education",when(df3.education=="unknown",np.nan).otherwise(df3.education))\
         .withColumn("education_na_ind",when(df3.education=="unknown",1).otherwise(0))\
         .withColumn("default",when(df3.default=="unknown",np.nan).otherwise(df3.default))\
         .withColumn("default_na_ind",when(df3.default=="unknown",1).otherwise(0))\
         .withColumn("housing",when(df3.housing=="unknown",np.nan).otherwise(df3.housing))\
         .withColumn("housing_na_ind",when(df3.housing=="unknown",1).otherwise(0))\
         .withColumn("loan",when(df3.loan=="unknown",np.nan).otherwise(df3.loan))\
         .withColumn("load_na_ind",when(df3.loan=="unknown",1).otherwise(0))\
         .withColumn("y_ind",when(df3.pdays=="no",0).otherwise(1))

#####################
## feature selection
#####################
categorical_features = [t[0] for t in df4.dtypes if t[1] == 'string' and t[0] != "y"]
numeric_features = [t[0] for t in df4.dtypes if t[1] != 'string' and t[0] != 'wasPreviouslyContacted' and t[0]!='y_int']

#####################
## pipeline
#####################

indexers = [
    StringIndexer(inputCol= col, outputCol="{0}_indexed".format(col))
    for col in categorical_features
]

y_indexer = [StringIndexer(inputCol= "y",
                          outputCol="label")]

encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=[
        "{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
)

assembler = VectorAssembler(
    inputCols=encoder.getOutputCols()+numeric_features,
    outputCol="vectorized_features"
)

scaler = StandardScaler()\
         .setInputCol ("vectorized_features")\
         .setOutputCol ("scaled_features")
         #.setHandleInvalid("skip")

pipeline = Pipeline(stages=indexers + y_indexer + [encoder, assembler, scaler])
df5 = pipeline.fit(df4).transform(df4)

#####################
# Export data to file
#####################


#####################
# Build Models
#####################

# data split
train, test = df5.randomSplit([0.8, 0.2], seed = 2018)

# Model1
lr = LogisticRegression(featuresCol = 'scaled_features', labelCol = 'label', maxIter=5)

lrModel = lr.fit(train)

predictions = lrModel.transform(test)

# Model1 summary 
trainingSummary = lrModel.summary
trainingSummary.accuracy
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# model1 parameter tuning
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])# regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])# Elastic Net Parameter (Ridge = 0)
             .addGrid(lr.maxIter, [1, 5, 10])#Number of iterations
             .build())

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, numFolds=5)

cvModel = cv.fit(train)

predictions = cvModel.transform(test)
print('Best Model Test Area Under ROC', evaluator.evaluate(predictions))

weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
weightsDF.toPandas().head(10)

best_model=cvModel.bestModel

#####################
# Model to file
#####################
lrModel.save("../models/_best/lrModel")