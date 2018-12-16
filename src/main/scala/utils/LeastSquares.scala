package utils

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors, Vector => SVector}

import algebra.MatrixOps._
import algebra.VectorOps._

object LeastSquares extends LossFunction {
  def loss(x: SVector, y: SVector): Double = {
    assert(x.size == y.size)
    x.toArray.zip(y.toArray).map { case (a, b) => (a - b) * (a - b) }.sum / (2 * x.size)
  }

  def grad(x: RowMatrix, y: SVector, beta: SVector): SVector = {
    beta.dotMatrix(x).minus(y).dotMatrix(x)
  }
}