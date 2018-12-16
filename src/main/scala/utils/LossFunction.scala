package utils

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors, Vector => SVector}

trait LossFunction {
  def loss(x: SVector, y: SVector): Double
  def grad(x: RowMatrix, y: SVector, beta: SVector): SVector
}
