import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

class MNISTUtils {

    public static class MNISTData {
        public double[][][] images;
        public double[][] labels;

        public MNISTData(double[][][] images, double[][] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static MNISTData loadMNISTFromFiles(String imagesPath, String labelsPath, int limit) {
        try {
            byte[] imageData = Files.readAllBytes(Paths.get(imagesPath));
            byte[] labelData = Files.readAllBytes(Paths.get(labelsPath));

            int imageMagic = readInt(imageData, 0);
            int labelMagic = readInt(labelData, 0);

            if (imageMagic != 2051 || labelMagic != 2049) {
                throw new RuntimeException("Invalid MNIST files: incorrect magic numbers.");
            }

            int numImages = readInt(imageData, 4);
            int numRows = readInt(imageData, 8);
            int numCols = readInt(imageData, 12);
            int numLabels = readInt(labelData, 4);

            System.out.printf("MNIST: %d images, %dx%d pixels%n", numImages, numRows, numCols);

            int actualLimit = Math.min(limit, numImages);
            // Images are [limit][height][width]
            double[][][] images = new double[actualLimit][numRows][numCols];
            // Labels are one-hot encoded: [limit][10]
            double[][] labels = new double[actualLimit][10];

            int imageOffset = 16;
            int labelOffset = 8;

            for (int i = 0; i < actualLimit; i++) {
                // Read image data
                for (int row = 0; row < numRows; row++) {
                    for (int col = 0; col < numCols; col++) {
                        int pixelIndex = imageOffset + i * (numRows * numCols) + row * numCols + col;
                        // Pixels are unsigned bytes (0-255), normalize to 0.0-1.0
                        images[i][row][col] = (imageData[pixelIndex] & 0xFF) / 255.0;
                    }
                }

                // Read label data and one-hot encode it
                int label = labelData[labelOffset + i] & 0xFF;
                for (int j = 0; j < 10; j++) {
                    labels[i][j] = (j == label) ? 1.0 : 0.0;
                }
            }

            return new MNISTData(images, labels);

        } catch (IOException e) {
            System.err.println("Error loading MNIST files: " + e.getMessage());
            System.err.println("Using simulated data instead...");
            return loadMNISTSimulated(limit);
        }
    }

    private static int readInt(byte[] data, int offset) {
        return ((data[offset] & 0xFF) << 24) |
                ((data[offset + 1] & 0xFF) << 16) |
                ((data[offset + 2] & 0xFF) << 8) |
                (data[offset + 3] & 0xFF);
    }

    public static MNISTData loadMNISTFromCSV(String csvPath, int limit, boolean hasHeader) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(csvPath));
            int startLine = hasHeader ? 1 : 0;
            int actualLimit = Math.min(limit, lines.size() - startLine);

            double[][][] images = new double[actualLimit][28][28];
            double[][] labels = new double[actualLimit][10];

            for (int i = 0; i < actualLimit; i++) {
                String[] values = lines.get(i + startLine).split(",");

                int label = Integer.parseInt(values[0]);
                for (int j = 0; j < 10; j++) {
                    labels[i][j] = (j == label) ? 1.0 : 0.0;
                }

                for (int pixel = 0; pixel < 784; pixel++) {
                    int row = pixel / 28;
                    int col = pixel % 28;
                    double pixelValue = Double.parseDouble(values[pixel + 1]) / 255.0;
                    images[i][row][col] = pixelValue;
                }
            }

            return new MNISTData(images, labels);

        } catch (IOException e) {
            System.err.println("Error loading CSV file: " + e.getMessage());
            return loadMNISTSimulated(limit);
        }
    }
    
    public static MNISTData loadMNISTSimulated(int limit) {
        System.out.println("⚠️ Using simulated MNIST data for demonstration.");
        Random random = new Random(42);

        double[][][] images = new double[limit][28][28];
        double[][] labels = new double[limit][10];

        for (int i = 0; i < limit; i++) {
            // Create some basic patterns with noise
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    double noise = random.nextGaussian() * 0.1;
                    double pattern = Math.sin(j * 0.3) * Math.cos(k * 0.3) + 0.5;
                    images[i][j][k] = Math.max(0, Math.min(1, pattern + noise));
                }
            }

            int labelIndex = random.nextInt(10);
            for (int j = 0; j < 10; j++) {
                labels[i][j] = (j == labelIndex) ? 1.0 : 0.0;
            }
        }
        return new MNISTData(images, labels);
    }
    
    public static MNISTData loadMNIST(int limit) {
        if (Files.exists(Paths.get("train-images-idx3-ubyte")) &&
                Files.exists(Paths.get("train-labels-idx1-ubyte"))) {
            System.out.println("📁 Loading from MNIST binary files...");
            return loadMNISTFromFiles("train-images-idx3-ubyte", "train-labels-idx1-ubyte", limit);
        }

        if (Files.exists(Paths.get("mnist/train-images-idx3-ubyte")) &&
                Files.exists(Paths.get("mnist/train-labels-idx1-ubyte"))) {
            System.out.println("📁 Loading from mnist/ directory...");
            return loadMNISTFromFiles("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", limit);
        }

        if (Files.exists(Paths.get("train.csv"))) {
            System.out.println("📁 Loading from train.csv (Kaggle format)...");
            return loadMNISTFromCSV("train.csv", limit, true);
        }

        return loadMNISTSimulated(limit);
    }

    public static double[] flatten(double[][][] matrix) {
        int depth = matrix.length;
        int height = matrix[0].length;
        int width = matrix[0][0].length;
        double[] result = new double[depth * height * width];

        int index = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[index++] = matrix[d][h][w];
                }
            }
        }
        return result;
    }

    public static double[][][] reshape(double[] vector, int depth, int height, int width) {
        if (vector.length != depth * height * width) {
            throw new IllegalArgumentException("Vector length does not match target dimensions.");
        }
        double[][][] result = new double[depth][height][width];
        int index = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[d][h][w] = vector[index++];
                }
            }
        }
        return result;
    }

    public static int argmax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}