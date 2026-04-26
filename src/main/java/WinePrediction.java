import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WinePrediction {
    public static void main(String[] args) throws Exception {

        if (args.length < 1) {
            System.err.println("Usage: WinePrediction <test-csv-path>");
            System.exit(1);
        }

        String testFile = args[0];

        SparkSession spark = SparkSession.builder()
            .appName("WineQualityPrediction")
            .master("local[*]")
            .getOrCreate();

        Dataset<Row> rawTrain = spark.read()
            .option("header", "true")
            .option("sep", ";")
            .option("inferSchema", "true")
            .csv("/app/TrainingDataset.csv");

        Dataset<Row> trainData = rawTrain;
        for (String col : rawTrain.columns()) {
            trainData = trainData.withColumnRenamed(col, col.replaceAll("\"", "").trim());
        }
        trainData = trainData.withColumnRenamed("quality", "label");

        String[] featureCols = {
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH",
            "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureCols)
            .setOutputCol("features");

        RandomForestClassifier rf = new RandomForestClassifier()
            .setNumTrees(100)
            .setMaxDepth(10)
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setSeed(42);

        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{assembler, rf});

        System.out.println("=== Training Model ===");
        PipelineModel model = pipeline.fit(trainData);
        System.out.println("=== Model Ready ===");

        Dataset<Row> rawTest = spark.read()
            .option("header", "true")
            .option("sep", ";")
            .option("inferSchema", "true")
            .csv(testFile);

        Dataset<Row> testData = rawTest;
        for (String col : rawTest.columns()) {
            testData = testData.withColumnRenamed(col, col.replaceAll("\"", "").trim());
        }
        testData = testData.withColumnRenamed("quality", "label");

        Dataset<Row> predictions = model.transform(testData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);

        System.out.println("==============================");
        System.out.println("F1 Score: " + f1);
        System.out.println("==============================");

        spark.stop();
    }
}
