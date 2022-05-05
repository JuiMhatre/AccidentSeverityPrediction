

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.sql._

import scala.collection.mutable.ListBuffer

object Accident{
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("Accident")
      .config("spark.master", "local[2]")
      .getOrCreate();
    val sc = spark.sparkContext
    val filePath = "C:\\Users\\IdeaProjects\\BigDataHW\\data\\US_Accidents_Dec20_updated_out.csv"
    val df =spark.read.format("csv").option("header","true")
      .option("nullValue","null").option("treatEmptyValuesAsNulls,","true").load(filePath)
    print(df.show())
    var data = sc.textFile(filePath)
    val header = data.first()
    data = data.filter(row => row !=header)
    val rdd =data.map(line => {
      val part =line.split(',')
      LabeledPoint(part(1).toDouble-1,
        Vectors.dense(part.slice(2,36).map(_.toDouble)))
    }).cache()

    val splits = rdd.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
// Decision Trees------------------Start---------------------

    val numClasses = 4
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val modeldt = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val metricsdt = getMetrics(modeldt, testData)
    println("Decision Tree Accuracy "+ metricsdt.accuracy)
    println("Decision Tree CM "+ metricsdt.confusionMatrix)
    val pred_label_DT = testData.map { point =>
      val prediction = modeldt.predict(point.features)
      ( prediction,point.label)
    }
    pred_label_DT.first()
    // Decision Trees------------------End---------------------




    // Logistic Regression------------------Start---------------------
    val modellr = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(trainingData)
    val pred_labelLR = testData .map { case LabeledPoint(label, features) =>
      val prediction = modellr.predict(features)
      (prediction,label)
    }
    val metricslr = new MulticlassMetrics(pred_labelLR)
    println("Logistic Regression Accuracy "+ metricslr.accuracy)
    println("Logistic Regression CM "+ metricslr.confusionMatrix)
    // Logistic Regression------------------End-----------------------



    // KNN------------------Start---------------------
    val array_train = trainingData.collect()
    val pred_labelKNN = testData .map { case lbdpt =>
      val prediction = predictClassification(array_train, lbdpt,1)
      (prediction,lbdpt.label)
    }
    val metricsknn = new MulticlassMetrics(pred_labelKNN)
    println("KNN Accuracy "+ metricsknn.accuracy)
    println("KNN CM "+ metricsknn.confusionMatrix)

    // KNN------------------End---------------------
    val min_max_pred = ListBuffer[(Double,Double)]()
    for( i <- 1 to pred_labelLR.collect().length){
      val predicted = if (pred_label_DT.collect().slice(i,i)(1)._2 == pred_labelKNN.collect().slice(i,i)(1)._2)
                          pred_label_DT.collect().slice(i,i)(1)._2 else
                          pred_labelLR.collect().slice(i,i)(1)._2

      min_max_pred +=(pred_labelLR.collect().slice(i,i)(1)._1,predicted)

    }
    val min_max_rdd = sc.parallelize(min_max_pred.toSeq)
    val metricsEL = new MulticlassMetrics(min_max_rdd)
    println("Ensemble Learning Accuracy "+ metricsEL.accuracy)
    println("Ensemble Learning CM "+ metricsEL.confusionMatrix)
  }
  def predictClassification(trainingData: Array[LabeledPoint], testRow: LabeledPoint, k: Int): Double ={
    getNeighbours(trainingData, testRow, k)
  }
  def computeEuclideanDistance(row1: LabeledPoint, row2: LabeledPoint): Double = {
    var distance = 0.0
    for (i <- 0 until row1.features.toSparse.size - 1) {
      distance += math.pow(row1.getFeatures(i) - row2.getFeatures(i), 2)
    }

    math.sqrt(distance)
  }
  def getNeighbours(trainSet: Array[LabeledPoint], testRow: LabeledPoint, k: Int): Double = {
    var distances = ListBuffer[(LabeledPoint,Double)]()
    trainSet.foreach(trainRow =>{
      val dist = computeEuclideanDistance(trainRow,testRow)
      val x = (trainRow,dist)
      distances += x
    })

    distances = distances.sortBy(_._2)
    distances(1)._1.label
  }
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

}
