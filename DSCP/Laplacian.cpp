#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace cv;
using namespace std;



void image_preprocessing(Mat& image, Mat& preprocessed_image);
void saveImage(const string& filename, const Mat& image);
Mat loadImage(const string& filename);

void applyLaplacianSharpen(const Mat& inputChannel, Mat& outputChannel, double sharpeningFactor) {
    Mat laplacian, blurred;


    //GaussianBlur(inputChannel, blurred, Size(3, 3), 0);

    Laplacian(inputChannel, laplacian, CV_16S, 3);
    convertScaleAbs(laplacian, laplacian);

    /*Mat laplacianThresholded;
    threshold(laplacian, laplacianThresholded, 10, 255, THRESH_TOZERO);*/


    addWeighted(inputChannel, 1.0, laplacian, sharpeningFactor, 0, outputChannel);
}

void applyLaplacianToColorImage(Mat& inputImage, Mat& outputImage, double sharpeningFactor) {
    Mat preProcessed;
    vector<Mat> channels(3);



    split(inputImage, channels);

    vector<Mat> outputChannels(3);
    for (int i = 0; i < 3; ++i) {
        image_preprocessing(channels[i], preProcessed);
        applyLaplacianSharpen(channels[i], outputChannels[i], sharpeningFactor);
    }

    // Merge the processed channels back into a color image
    merge(outputChannels, outputImage);
}

Mat single_Laplacian_Filter(Mat& image) {

    //image = loadImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/IC.png");
    
    


    Mat sharpenedColor;
    // Apply the Laplacian filter
    applyLaplacianToColorImage(image, sharpenedColor, 0.2);

    saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Single.png", sharpenedColor);

    return sharpenedColor;

}

Mat mpi_Laplacian_Filter(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int rows, cols, channels;
    int blockSize;
    vector<uchar> imageData;

    if (world_rank == 0) {
        // Load the image as a color image
        string imagePath;
        cin.ignore();
        cout << "Image Path: ";
        getline(cin, imagePath);

        Mat image = loadImage(imagePath);
        if (image.empty()) {
            cerr << "Error: Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Ensure the image is in 3 channels (BGR)
        if (image.channels() != 3) {
            cout << "Error: Input image does not have 3 channels!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
        blockSize = rows / world_size;

        // Flatten image data for distribution
        imageData.resize(rows * cols * channels);
        memcpy(imageData.data(), image.data, imageData.size());
    }

    // Broadcast image dimensions and block size
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter image data to all processes
    vector<uchar> localData(blockSize * cols * channels);
    MPI_Scatter(imageData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);


    Mat localBlock(blockSize, cols, CV_8UC3, localData.data());


    Mat localOutputBlock;
    double sharpeningFactor = 0.2;  // Adjust sharpening intensity (0.3 to 0.7 for subtle or stronger effects)
    applyLaplacianToColorImage(localBlock, localOutputBlock, sharpeningFactor);

    // Gather processed blocks
    vector<uchar> resultData(rows * cols * channels);
    MPI_Gather(localOutputBlock.data, blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        resultData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);
    Mat resultImage;

    if (world_rank == 0) {
        // Reconstruct the image from result data
        resultImage = Mat(rows, cols, CV_8UC3, resultData.data());

        // Save the final sharpened image
        saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Laplacian.png", resultImage);
    }

    MPI_Finalize();
    return resultImage;
}