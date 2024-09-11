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

void unsharpMasking(const Mat& originalImageBlock, const Mat& preprocessedBlock, Mat& outputBlock) {
    Mat blurred, mask, sharpened;

    // Apply Gaussian Blur to the preprocessed image block to get a blurred version
    //GaussianBlur(preprocessedBlock, blurred, Size(3, 3), 1.0);

    // Create the mask by subtracting the blurred image from the original image block
    subtract(originalImageBlock, blurred, mask);

    // Sharpen the original image by adding the mask to the original image
    add(originalImageBlock, mask, sharpened);

    // Set the output block
    outputBlock = sharpened;
}

Mat single_thread_unsharp_masking(Mat& image) {

    Mat preProcessed;

    image_preprocessing(image, preProcessed);


    Mat sharpenedImage;
    unsharpMasking(image, preProcessed,sharpenedImage);


    // Save the result
    saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Single.png", sharpenedImage);

    return preProcessed;
}


Mat mpi_unsharp_masking(int argc, char** argv, Mat& image) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Mat preprocessedImage;
    int rows, cols, channels;
    int blockSize;
    vector<uchar> imageData, preprocessedData;

    if (world_rank == 0) {
        // Load the image
        image = loadImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/IC.png");
        if (image.empty()) {
            cerr << "Error: Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Preprocess the image
        image_preprocessing(image, preprocessedImage);

        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
        blockSize = rows / world_size;

        // Flatten image data for distribution
        imageData.resize(rows * cols * channels);
        memcpy(imageData.data(), image.data, imageData.size());

        // Flatten preprocessed image data
        preprocessedData.resize(rows * cols * channels);
        memcpy(preprocessedData.data(), preprocessedImage.data, preprocessedData.size());
    }

    // Broadcast image dimensions and block size
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter image data and preprocessed image data to all processes
    vector<uchar> localData(blockSize * cols * channels);
    vector<uchar> localPreprocessedData(blockSize * cols * channels);
    MPI_Scatter(imageData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);
    MPI_Scatter(preprocessedData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localPreprocessedData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Create local image block and preprocessed block
    Mat localBlock(blockSize, cols, CV_8UC3, localData.data());
    Mat localPreprocessedBlock(blockSize, cols, CV_8UC3, localPreprocessedData.data());

    // Allocate space for output block
    Mat localOutputBlock(blockSize, cols, CV_8UC3);

    // Apply unsharp masking on the original and preprocessed image blocks
    unsharpMasking(localBlock, localPreprocessedBlock, localOutputBlock);

    // Gather processed blocks
    vector<uchar> resultData(rows * cols * channels);
    MPI_Gather(localOutputBlock.data, blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        resultData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    Mat resultImage;

    if (world_rank == 0) {
        // Reconstruct and save the final sharpened image
        resultImage = Mat(rows, cols, CV_8UC3, resultData.data());
        saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Unsharp_Masking.png", resultImage);
    }

    MPI_Finalize();
    return resultImage;
}





