import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;

public class TimeSeriesForecast {
    private final double[] timeSeriesData;  // 时间序列数据
    private final int predictionLength;    // 预测长度

    public TimeSeriesForecast(double[] timeSeriesData, int predictionLength) {
        this.timeSeriesData = timeSeriesData;
        this.predictionLength = predictionLength;
    }

    // 差分
    private double[] diff(double[] data) {
        double[] diffData = new double[data.length - 1];
        for (int i = 1; i < data.length; i++) {
            diffData[i - 1] = data[i] - data[i - 1];
        }
        return diffData;
    }

    // 逆差分
    private double[] invDiff(double[] diffData, double initValue) {
        double[] data = new double[diffData.length + 1];
        data[0] = initValue;
        for (int i = 1; i < data.length; i++) {
            data[i] = data[i - 1] + diffData[i - 1];
        }
        return data;
    }

    // 构建x矩阵
    private double[][] buildXMatrix(double[] x) {
        double[][] xMatrix = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            xMatrix[i][0] = 1;  // 第一列全为1
            xMatrix[i][1] = x[i];
        }
        return xMatrix;
    }

    // 构建回归模型
    private OLSMultipleLinearRegression buildRegressionModel(double[] x, double[] y) {
        OLSMultipleLinearRegression regressionModel = new OLSMultipleLinearRegression();
        regressionModel.newSampleData(y, buildXMatrix(x));
        return regressionModel;
    }

    // 使用回归模型进行预测
    private double predict(double[] x, OLSMultipleLinearRegression regressionModel) {
        RealMatrix coefficients = MatrixUtils.createColumnRealMatrix(regressionModel.estimateRegressionParameters());
        RealMatrix xMatrix = MatrixUtils.createRowRealMatrix(x);
        return coefficients.transpose().multiply(xMatrix.transpose()).getEntry(0, 0);
    }

    // 生成预测数据
    private double[] generateForecast(double[] diff, int p) {
        double[] forecast = new double[predictionLength];
        double[] x = Arrays.copyOfRange(diff, diff.length - p, diff.length);
//        OLSMultipleLinearRegression regressionModel = buildRegressionModel(x, Arrays.copyOfRange(diff, diff.length - p + 1, diff.length));
        OLSMultipleLinearRegression regressionModel = buildRegressionModel(x, Arrays.copyOfRange(diff, diff.length - p, diff.length - 1));

        for (int i = 0; i < predictionLength; i++) {
            double y = predict(x, regressionModel);
            forecast[i] = y + timeSeriesData[timeSeriesData.length - 1];
            // 更新x向量
            for (int j = p - 1; j >= 1; j--) {
                x[j] = x[j - 1];
            }
            x[0] = forecast[i] - timeSeriesData[timeSeriesData.length - p + i + 1];
        }
        return forecast;
    }

    // 进行时间序列预测
    public double[] forecast(int p) {
        double[] diffData = diff(timeSeriesData);
        return invDiff(generateForecast(diffData, p), timeSeriesData[timeSeriesData.length - 1]);
    }

    public static void main(String[] args) {
        // 测试数据
//        double[] data = {10.1, 11.2, 12.0, 11.8, 13.2, 13.9, 13.6, 15.2, 17.5, 19.4, 21.0, 23.2, 24.4, 25.8};
//        double[] data = {10.11, 11.21, 12.01, 11.81, 13.2, 13.91, 13.6, 15.21, 17.5, 19.4, 21.01, 23.21, 24.41, 25.8};
//        double[] data = {11.11, 11.21, 12.01, 11.81, 13.2, 13.91, 12.6, 15.21, 14.5, 19.4, 21.01, 23.21, 24.41, 25.8};
//        double[] data = {11.31, 11.21, 12.01, 11.81, 13.2, 13.91, 12.6, 15.21, 14.5, 19.4, 21.01, 23.21, 24.41, 25.8};
        double[] data = {11.31, 11.21, 13.01, 11.81, 13.2, 13.91, 12.6, 15.21, 14.5, 19.4, 21.01, 23.21, 24.41, 25.8};
        int predictionLength = 5;
        int p = 2;
        TimeSeriesForecast tsf = new TimeSeriesForecast(data, predictionLength);
        double[] forecast = tsf.forecast(p);
        System.out.println("原始数据：" + Arrays.toString(data));
        System.out.println("预测结果：" + Arrays.toString(forecast));
    }

}