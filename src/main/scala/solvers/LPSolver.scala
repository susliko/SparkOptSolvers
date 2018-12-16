package solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector => SVector}

object LPSolver {
  import algebra.MatrixOps._
  import algebra.VectorOps._
  case class Iteration(num: Int, x: SVector)

  def solveLP(A: RowMatrix, b: SVector, c: SVector, x0: SVector, gamma: Double = 0.5)(implicit sc: SparkContext) = {
    lazy val iterations: Stream[Iteration] = Iteration(0, x0) #:: iterations.map { it =>
      val X = DenseMatrix.diag(it.x)
      val doubleX = X.multiply(X)
      val p = A.multiply(doubleX).multiply(A.transpose.toLocal).inverse.multiply(A.toLocal).multiply(doubleX).dotVector(c)
      val r = c.minus(A.transpose.dotVector(p))
      val norm = math.sqrt(X.toRowMatrix.dotVector(r).toArray.map(x => x * x).sum)
      Iteration(it.num + 1, it.x.minus(doubleX.toRowMatrix.dotVector(r).dotScalar(gamma / norm)))
    }
    iterations
  }
}