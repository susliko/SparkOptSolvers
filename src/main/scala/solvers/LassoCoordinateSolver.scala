package solvers

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors, Vector => SVector}
import algebra.MatrixOps._
import algebra.VectorOps._
import utils.LossFunction

object LassoCoordinateSolver {
  def solve(lossFunction: LossFunction, x: RowMatrix, y: SVector, lambda: Double, epsilon: Double = 0.0001): SVector = {
    val k = x.numRows().toInt
    var beta = Vectors.dense((for (_ <- 0 until k) yield 0.0).toArray)
    var oldBeta = Vectors.dense((for (_ <- 0 until k) yield Double.PositiveInfinity).toArray)
    while (beta.minus(oldBeta).toArray.map(Math.abs).max > epsilon) {
      val g = lossFunction.grad(x, y, beta)
      val minIndex = g.toArray.flatMap(x => Vector(x, -x)).zipWithIndex.min._2
      val eMin = minIndex / 2
      val sign = if (minIndex % 2 == 0) 1 else -1
      oldBeta = beta
      beta = (for (i <- 1 to 1000) yield {
        val a = i / 1000.0
        val aBeta = beta.dotScalar(a)
        val newBeta = Vectors.dense(aBeta.toArray.updated(eMin, aBeta(eMin) + sign*(1 - a) * lambda))
        (lossFunction.loss(x.dotVector(newBeta), y), newBeta)
      }).minBy(_._1)._2
    }
    beta
  }
}
