package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => bVector}

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.sparkproject.dmg.pmml.gaussian_process.Lambda

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String, val numOfFeatures: Int, val lambda: Double,
                       val numOfIterations: Int) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this(numOfFeatures: Int, lambda: Double = 1,
           numOfIterations: Int = 1000) = this(Identifiable.randomUID("LinearRegression"),
                                             numOfFeatures, lambda, numOfIterations)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[(Double, Vector)] = ExpressionEncoder()

    var w: bVector[Double] = bVector.rand(numOfFeatures)
    var b = scala.util.Random.nextDouble()

    val vectors: Dataset[(Double, Vector)] = dataset.select(dataset($(inputCol)).as[(Double, Vector)])

    var i = 0
    for (i <- 1 to numOfIterations) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[(Double, Vector)]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val target = v._1
          val features = v._2.asBreeze
          val pred = features.t * w + b
          val error = (pred - target) * lambda
          val grad = error * features
          summarizer.add(mllib.linalg.Vectors.dense(Array(error) ++ grad.toArray))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      val shift = summary.mean.asML.asBreeze.toDenseVector
      val b_shift = shift(0).asInstanceOf[Double]
      val w_shift = shift(1 to -1)

      b = b - b_shift
//      println(b)
      w = w - w_shift
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w), b)).setParent(this)
    }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val coeffs: DenseVector, val bias: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(coeffs: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), coeffs.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coeffs, bias))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
          (x : Vector) => {
            Vectors.dense(x.asBreeze.t * coeffs.asBreeze + bias)
          })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = coeffs.asInstanceOf[Vector] -> bias.asInstanceOf[Vector]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val (coeffs, bias) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel(coeffs.asInstanceOf[DenseVector], bias.asInstanceOf[Int])
      metadata.getAndSetParams(model)
      model
    }
  }
}
