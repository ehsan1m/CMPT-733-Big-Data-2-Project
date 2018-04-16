from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier,LinearSVC
import ast
from pyspark.ml import Pipeline
from pyspark.sql import functions as F, types
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

spark = SparkSession \
    .builder \
    .appName("toxicity_classification") \
    .getOrCreate()
sc = spark.sparkContext



dfTrain = spark.read.csv('Data/FeaturizedDataTraining.csv',header=True)
column_names = dfTrain.schema.names[1:]
dfTrain = dfTrain.select(*(F.col(c).cast("float").alias(c) for c in column_names))
dfTrain = dfTrain.withColumn("label", dfTrain["label"].cast(types.IntegerType()))
dfVal = spark.read.csv('Data/FeaturizedDataValidation.csv',header=True)
column_names = dfVal.schema.names[1:]
dfVal = dfVal.select(*(F.col(c).cast("float").alias(c) for c in column_names))
dfVal = dfVal.withColumn("label", dfVal["label"].cast(types.IntegerType()))

dfTrain = dfTrain.withColumn('label',(dfTrain['label']>0).cast('integer'))
dfVal = dfVal.withColumn('label',(dfVal['label']>0).cast('integer'))

#Creating the data frame containing the training data
#pandas_df = pd.read_csv('Data/FeaturizedDataTraining.csv')
#pandas_df = pd.read_csv('Data/FeaturizedDataTrainingSample.csv')
#dfTrain = spark.createDataFrame(pandas_df)

#Creating the data frame containing the testing data
#pandas_df = pd.read_csv('Data/FeaturizedDataValidation.csv')
#pandas_df = pd.read_csv('Data/FeaturizedDataValidationSample.csv')
#dfVal = spark.createDataFrame(pandas_df)

#Creating the data frame containing the examples
pandas_df = pd.read_csv('Data/Examples.csv')
dfExample = spark.createDataFrame(pandas_df)

column_names = dfExample.schema.names[1:501] #column_names is the same for al three dfs

vec_assemb = VectorAssembler(inputCols=column_names, outputCol="Vecfeatures")
vec_assemb.transform(dfTrain)

rf = RandomForestClassifier(labelCol="label", featuresCol="Vecfeatures", numTrees=100)
lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol="Vecfeatures")
pipeline = Pipeline(stages=[vec_assemb,rf ])
ML_model = pipeline.fit(dfTrain)
predictions = ML_model.transform(dfVal)

#evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
#recall = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)


TP = predictions.where(predictions.label==1).where(predictions.prediction==1).count()
TN = predictions.where(predictions.label==0).where(predictions.prediction==0).count()
FP = predictions.where(predictions.label==0).where(predictions.prediction==1).count()
FN = predictions.where(predictions.label==1).where(predictions.prediction==0).count()
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('accuracy',accuracy)
print('recall:',recall)
print('precision',precision)

sample_pred = ML_model.transform(dfExample)

sample_pred.select('sentence', 'prediction').show()
ML_model.save("SavedSVMModel_rf_nt_100")


# dfTrain = spark.read.csv('Data/FeaturizedDataTraining.csv',header=True)
# column_names = dfTrain.schema.names[1:]
# dfTrain = dfTrain.select(*(F.col(c).cast("float").alias(c) for c in column_names))
# dfTrain = dfTain.withColumn("label", dfTrain["label"].cast(types.IntegerType()))

# dfTain = spark.read.csv('Data/FeaturizedDataTraining.csv',header=True)
# column_names = dfTain.schema.names[1:]
# dfTain = dfTain.select(*(F.col(c).cast("float").alias(c) for c in column_names))
# dfTain = dfTain.withColumn("label", dfTain["label"].cast(types.IntegerType()))

