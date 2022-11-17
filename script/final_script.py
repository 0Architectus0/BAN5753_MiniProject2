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
df2 = df.withColumnRenamed("cons.price.idx","cons_price_idx")\
.withColumnRenamed("emp.var.rate", "emp_var_rate")\
.withColumnRenamed("cons.conf.idx","cons_conf_idx")\
.withColumnRenamed("nr.employed", "nr_employed")

#####################
## feature selection
#####################
categorical_features = [t[0] for t in df2.dtypes if t[1] == 'string' and t[0] != "y"]
numeric_features = [t[0] for t in df2.dtypes if t[1] != 'string']
response_feature = ["y"]
#####################
## pipeline
#####################

indexers = [
    StringIndexer(inputCol= col,
                  outputCol="{0}_indexed".format(col))
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

pipeline = Pipeline(stages=indexers + y_indexer +
                    [encoder, assembler, scaler])
df3 = pipeline.fit(df2).transform(df2)

#####################
# Build Model - Null Model
#####################

# data split
train, test = df2.randomSplit([0.8, 0.2], seed = 2018)

# Model1
lr = LogisticRegression(featuresCol = 'scaled_features',
                        labelCol = 'label', maxIter=5)

lrModel = lr.fit(train)

predictions = lrModel.transform(test)

# Model_initial summary 
trainingSummary = lrModel.summary
trainingSummary.accuracy
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

#####################
# Tune Model - Null Model
#####################
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

best_model=cvModel.bestModel

#####################
# Model to file
#####################
lrModel.save("../models/lrModel1")

###############################################################
# Evaluate Model - Model1
###############################################################
training1Summary = lrModel.summary

training1Summary.accuracy
training1Summary.labels
training1Summary.recallByLabel
training1Summary.weightedFalsePositiveRate

# high model accuracy but low recall given the imbalanced data set

###############################################################
# Transform: Oversample Data
###############################################################
from pyspark.sql.functions import col, asc, desc

major_df = df3.filter(col('label') == 0.0)
minor_df = df8.filter(col("label") == 1.0)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))
a = range(ratio)

from pyspark.sql.functions import col, explode, array, lit
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy",
                                     explode(array([lit(x) for x in a])))
                        .drop('dummy')

# combine both oversampled minority rows and previous majority rows 
combined_df = major_df.unionAll(oversampled_df)
###############################################################
# Build Model - Model2 - Oversampled minority class
###############################################################
lr2 = LogisticRegression(featuresCol = 'scaled_features',
                         labelCol = 'label', maxIter=5)

lrModel2 = lr2.fit(train_over)

predictions_os = lrModel2.transform(test)
mod2_accuarcy = lrModel2.summary.accuracy
mod2_recallByLabel = lrModel2.summary.recallByLabel

#predictions2 = lrModel2.transform(test_over)

#####################
# Tune Model2
#####################
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])# regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])# Elastic Net Parameter (Ridge = 0)
             .addGrid(lr.maxIter, [1, 5, 10])#Number of iterations
             .build())

cv = CrossValidator(estimator=lr2, estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, numFolds=5)

cvModel = cv.fit(train_over)

predictions_cv_os = cvModel.transform(test)
print('Best Model Test Area Under ROC', evaluator.evaluate(predictions))

best_model_oversampled = cvModel.bestModel

#####################
# Model to file
#####################
best_model_oversampled.save("../models/lrModel2")

###############################################################
# Evaluate Model - Model2 - Oversampled minority class
###############################################################

predictions2.select('label', 'scaled_features', 'rawPrediction',
                    'prediction', 'probability').toPandas().head(5)

# in the earlier example
training2Summary = lrModel2.summary

training2Summary.accuracy
training2Summary.labels
training2Summary.recallByLabel
training2Summary.weightedFalsePositiveRate

###############################################################
# Graph Model - Model2, Confusion Matrix
###############################################################
y_true = predictions2.select("label")
y_true = y_true.toPandas()

y_pred = predictions2.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

