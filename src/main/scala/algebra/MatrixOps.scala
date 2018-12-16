package algebra

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Vectors, Vector => SVector}

object MatrixOps {

  implicit class DenseMatrixOps(m: DenseMatrix) {
    def toRowMatrix(implicit sc: SparkContext) = {
      new RowMatrix(sc.parallelize(m.values.grouped(m.numCols).map(Vectors.dense).toSeq))
    }
  }

  implicit class RowMatrixOps(m: RowMatrix) {
    def inverse(implicit sc: SparkContext): RowMatrix = {
      val nCoef = m.numCols.toInt
      val svd = m.computeSVD(nCoef, computeU = true)
      if (svd.s.size < nCoef) {
        sys.error(s"RowMatrix.computeInverse called on singular matrix.")
      }

      val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
      val U = new DenseMatrix(svd.U.numRows().toInt, svd.U.numCols().toInt, svd.U.rows.collect.flatMap(x => x.toArray))
      val V = svd.V
      new RowMatrix(sc.parallelize(
        V.multiply(invS)
          .multiply(U)
          .values
          .grouped(m.numRows().toInt)
          .toSeq
          .map(Vectors.dense))).transpose
    }

    def dotVector(v: SVector): SVector =
      Vectors.dense(m.multiply(new DenseMatrix(v.size, 1, v.toArray)).rows.collect().flatMap(_.toArray))

    def toLocal: DenseMatrix = {
      val rows = m.rows.collect()
      val rowsNum = rows.length
      val columnsNum = rows(0).size
      new DenseMatrix(rowsNum, columnsNum, rows.flatMap(_.toArray), true)
    }

    def transpose(implicit sc: SparkContext): RowMatrix =
        new RowMatrix(sc.parallelize(m.toLocal.toArray.grouped(m.numRows().toInt).toArray.map(Vectors.dense)))

  }
}
