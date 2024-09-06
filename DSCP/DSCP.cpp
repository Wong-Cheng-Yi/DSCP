#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

void apply_morphology(Mat& image, Mat& output) {
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(image, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_CLOSE, element);
}

void process_block(Mat& block, Mat& output_block) {
    apply_morphology(block, output_block);
}

void image_preprocessing(Mat& image, string imagePath) {
    int rows = image.rows;
    int cols = image.cols;
    int block_size = 64;

    int num_threads = omp_get_max_threads();
    cout << "Number of threads available: " << num_threads << endl;

    omp_set_num_threads(num_threads);

    Mat result = image.clone();

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                int block_width = min(block_size, cols - j);
                int block_height = min(block_size, rows - i);

                Mat block = image(Rect(j, i, block_width, block_height));
                Mat output_block = block.clone();

                process_block(block, output_block);

#pragma omp critical
                {
                    output_block.copyTo(result(Rect(j, i, block_width, block_height)));
                }
            }
        }
    }

    imwrite("processed_" + imagePath + ".png", result);
}

int main() {
    string imagePath = "";
    cout << "Image Path: ";

    getline(cin, imagePath);

    Mat image = imread("Image\\" + imagePath, IMREAD_COLOR);

    if (image.empty()) {
        cout << "Could not load the image!" << endl;
        return -1;
    }

    // Perform image preprocessing using OpenMP
    //image_preprocessing(image, imagePath);

    // Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    moveWindow("Original Image", 0, 0);

    // Load and display the processed image
    /*Mat processed_image = imread("processed_" + imagePath + ".png");
    if (!processed_image.empty()) {
        namedWindow("Processed Image", WINDOW_AUTOSIZE);
        imshow("Processed Image", processed_image);
        moveWindow("Processed Image", 500, 0);
    }
    else {
        cout << "Could not load the processed image!" << endl;
    }*/

    waitKey(0); // Wait for a key press
    return 0;
}
