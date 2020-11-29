package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix => bMatrix, DenseVector => bVector}
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val df: Dataset[_] = LinearRegressionTest._df
  lazy val small_df: Dataset[_] = LinearRegressionTest._small_df

  "Model" should "scale input data" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coeffs = Vectors.dense(1, -0.5, 0.5),
      bias = 3.0
    ).setInputCol("data").setOutputCol("data")

    val vectors: Array[Vector] = model.transform(small_df).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be(4.5 +- delta)

    vectors(1)(0) should be(5.5 +- delta)
  }

  "Estimator" should "calculate coeffs" in {
    val estimator = new LinearRegression(3)
      .setInputCol("data")
      .setOutputCol("data")

    val model = estimator.fit(df)

    model.bias should be(3.0+- delta)
    model.coeffs(0) should be(0.5 +- delta)
    model.coeffs(1) should be(-1.0 +- delta)
    model.coeffs(2) should be(2.0 +- delta)
  }

//
//  "Estimator" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(Array(
//      new LinearRegression()
//        .setInputCol("features")
//        .setOutputCol("features")
//    ))
//
//    val tmpFolder = Files.createTempDir()
//
//    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[LinearRegressionModel]
//
//    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
//    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
//  }
//
//  "Model" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(Array(
//      new LinearRegression()
//        .setInputCol("features")
//        .setOutputCol("features")
//    ))
//
//    val model = pipeline.fit(data)
//
//    val tmpFolder = Files.createTempDir()
//
//    model.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
//
//    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
//  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _X: bMatrix[Double] = bMatrix.rand(10000, 3)
  lazy val _y: bVector[Double] = _X * bVector(0.5, -1, 2) + 3.0
//  lazy val _y: bVector[Double] = _X * bVector(0.0, 0.0, 0.0) + 3.0
  lazy val _data: bMatrix[Double] = bMatrix.horzcat(_y.asDenseMatrix.t, _X)

  lazy val _df: Dataset[_] = {
    import sqlc.implicits._
    _data(*, ::).iterator.map(x => {
      Tuple1(x(0), Vectors.dense(x(1), x(2), x(3)))
    }).toSeq.toDF("data")
  }

  lazy val _small_df: Dataset[_] = {
    import sqlc.implicits._
    Seq(Tuple1(Vectors.dense(1, 2, 3)),
        Tuple1(Vectors.dense(1, -1, 2))).toDF("data")
  }
}
