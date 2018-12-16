package examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import solvers.{LassoCoordinateSolver, LassoGradientSolver}
import utils.LeastSquares
import algebra.MatrixOps._
import algebra.VectorOps._

import scala.util.Random


object LassoExample extends App{
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val sc = new SparkContext(master = "spark://researcher:7077", appName = "test")

  val n = 10
  val k = 20
  val x = new RowMatrix(sc.parallelize(for (_ <- 0 until k) yield {
    Vectors.dense((for (_ <- 0 until n) yield 2*Random.nextDouble() - 1).toArray)
  }))
  val realBeta = Vectors.dense((for (_ <- 0 until k) yield 2*Random.nextDouble() - 1).toArray)
  val y = x.dotVector(realBeta)
  val lambda = 1.2*realBeta.toArray.map(Math.abs).sum

  println(realBeta)
  val beta1 = LassoCoordinateSolver.solve(LeastSquares, x, y, lambda)
  println(beta1)
  val beta2 = LassoGradientSolver.solve(LeastSquares, x, y, lambda)
  println(beta2)
}
