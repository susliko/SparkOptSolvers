package algebra

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Vector => SVector}


object VectorOps {
  import algebra.MatrixOps._

  implicit class SVOps(v1: SVector)  {
    def plus(v2: SVector) = Vectors.dense(v1.toArray.zip(v2.toArray).map(p => p._1 + p._2))
    def minus(v2: SVector) = Vectors.dense(v1.toArray.zip(v2.toArray).map(p => p._1 - p._2))
    def negate = Vectors.dense(v1.toArray.map(-_))
    def dotMatrix(m: RowMatrix): SVector =
      Vectors.dense(Matrices.dense(1, v1.size, v1.toArray).multiply(m.toLocal).values)
    def dotScalar(c: Double): SVector = Vectors.dense(v1.toArray.map(_ * c))
    def dotVector(v2: SVector): Double = v1.toArray.zip(v2.toArray).map(p => p._1 * p._2).sum
  }
}
