# SparkOptSolvers
Lasso and LP solvers for Apache Spark

**Linear programming**
```scala
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object LPExample extends App{
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  import solvers.LPSolver._

  implicit val sc = new SparkContext(master = "spark://gopher:7077", appName = "test")

  val A = new RowMatrix(sc.parallelize(Seq(
    Vectors.dense(1, 1),
    Vectors.dense(-1, 1),
    Vectors.dense(5, 4)
  )))
  val b = Vectors.dense(10, -3, 35)
  val c = Vectors.dense(5, 6)

  val res = solveLP(A, b, c, Vectors.dense(3.1, 0.05)).take(300).foreach(println)

}
```

**Lasso**
