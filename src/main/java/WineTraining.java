import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WineTraining {
    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
            .appName("WineQualityTraining")
            .getOrCreate();

        Dataset<Row> rawTrain = spark.read()
            .option("header", "true")
            .option("sep", ";")
            .option("inferSchema", "true")
            .csv("/home/ubuntu/wine-quality/TrainingDataset.csv");

        Dataset<Row> trainData = rawTrain;
        for (String col : rawTrain.columns()) {
            trainData = trainData.withColumnRenamed(col, col.replaceAll("\"", "").trim());
        }
        trainData = trainData.withColumnRenamed("quality", "label");

        Dataset<Row> rawVal = spark.read()
            .option("header", "true")
            .option("sep", ";")
            .option("inferSchema", "true")
            .csv("/home/ubuntu/wine-quality/ValidationDataset.csv");

        Dataset<Row> valData = rawVal;
        for (String col : rawVal.columns()) {
            valData = valData.withColumnRenamed(col, col.replaceAll("\"", "").trim());
        }
        valData = valData.withColumnRenamed("quality", "label");

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

        System.out.println("=== Starting Training ===");
        PipelineModel model = pipeline.fit(trainData);
        System.out.println("=== Training Complete ===");

        Dataset<Row> predictions = model.transform(valData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("==============================");
        System.out.println("Validation F1 Score: " + f1);
        System.out.println("==============================");

        model.write().overwrite().save("/tmp/wine-model");
        System.out.println("=== Model saved to /tmp/wine-model ===");

        spark.stop();
    }
}
