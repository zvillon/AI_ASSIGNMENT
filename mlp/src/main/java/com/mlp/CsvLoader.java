package com.mlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class DataPair {
    final double[][] features;
    final double[][] labels;

    DataPair(double[][] features, double[][] labels) {
        this.features = features;
        this.labels = labels;
    }
}

public class CsvLoader {
    public static DataPair loadCsvData(String filePath, int numClasses, int expectedFeatures, String delimiter,
            boolean hasHeader, int limit) throws IOException, NumberFormatException {
        System.out.printf("  Loading CSV data from: %s (Limit: %d, Header: %s)\n", filePath, limit, hasHeader);
        List<double[]> featureList = new ArrayList<>();
        List<double[]> labelList = new ArrayList<>();
        String line;
        int rowsRead = 0;
        int expectedColumns = expectedFeatures + 1;

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            if (hasHeader) {
                br.readLine();
            }

            while ((line = br.readLine()) != null && (limit <= 0 || rowsRead < limit)) {
                String[] values = line.split(delimiter);
                if (values.length != expectedColumns) {
                    System.err.printf("Warning: Skipping row %d. Expected %d columns, found %d.\n",
                            (hasHeader ? rowsRead + 2 : rowsRead + 1), expectedColumns, values.length);
                    continue;
                }

                int labelValue = Integer.parseInt(values[0].trim());
                if (labelValue < 0 || labelValue >= numClasses) {
                    System.err.printf("Warning: Skipping row %d. Label '%d' out of bounds [0, %d).\n",
                            (hasHeader ? rowsRead + 2 : rowsRead + 1), labelValue, numClasses);
                    continue;
                }
                double[] oneHotLabel = new double[numClasses];
                oneHotLabel[labelValue] = 1.0;
                labelList.add(oneHotLabel);

                double[] features = new double[expectedFeatures];
                for (int i = 0; i < expectedFeatures; i++) {
                    double pixelValue = Double.parseDouble(values[i + 1].trim());
                    features[i] = pixelValue / 255.0;
                }
                featureList.add(features);

                rowsRead++;
                if (rowsRead > 0 && rowsRead % 10000 == 0)
                    System.out.printf("    ...read %d data rows\n", rowsRead);
            }
        }

        if (featureList.isEmpty()) {
            throw new IOException("No valid data rows found in CSV file: " + filePath);
        }

        System.out.printf("  Finished reading %d data rows.\n", rowsRead);

        double[][] featureArray = featureList.toArray(new double[0][]);
        double[][] labelArray = labelList.toArray(new double[0][]);

        return new DataPair(featureArray, labelArray);
    }
}