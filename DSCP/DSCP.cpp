#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>
#include "matplotlibcpp.h"
#include <limits>
#include <chrono>
using namespace std::chrono;

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;
using namespace plt;

void mpi_unsharp_masking(int argc, char** argv);
void mpi_Laplacian_Filter(int argc, char** argv);
Mat single_Laplacian_Filter(Mat& image);
Mat single_thread_unsharp_masking(Mat& image);
Mat omp_unsharp_masking(Mat&image);


void apply_morphology(Mat& image, Mat& output) {
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(image, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_CLOSE, element);
}

void image_preprocessing(Mat& image, Mat& preprocessed_image) {
    int rows = image.rows;
    int cols = image.cols;
    int block_size = 64;

    int num_threads = omp_get_max_threads();
    //cout << "Number of threads available: " << num_threads << endl;

    omp_set_num_threads(num_threads);

    preprocessed_image = image.clone();

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                int block_width = min(block_size, cols - j);
                int block_height = min(block_size, rows - i);

                Mat block = image(Rect(j, i, block_width, block_height));
                Mat output_block = block.clone();

                apply_morphology(block, output_block);

#pragma omp critical
                {
                    output_block.copyTo(preprocessed_image(Rect(j, i, block_width, block_height)));
                }
            }
        }
    }
}

//void unsharpMasking(const Mat& imageBlock, Mat& mask, Mat& outputBlock) {
//    Mat blurred, sharpened;
//    GaussianBlur(imageBlock, blurred, Size(3, 3), 1.0);
//    subtract(imageBlock, blurred, mask);
//    add(imageBlock, mask, sharpened);
//    outputBlock = sharpened;
//}

Mat loadImage(const string& filename) {
    Mat image = imread("C:/Users/wongc/source/repos/DSCP/DSCP/Image/" + filename, IMREAD_COLOR);
    return image;
}

// Function to save an image
void saveImage(const string& filename, const Mat& image) {
    imwrite(filename, image);
}



int main(int argc, char** argv) {

    //int choice;
    //int method;
    //Mat processed_image;
    //bool flag = false;

    //do {
    //    cout << "1:Laplacian Filter 2:Unsharp Masking\nWhat algorithm is used:";
    //    while (!(cin >> choice)) {  // Check if the input is of the correct type (int in this case)
    //        cin.clear();  // Clear the error flag on cin
    //        cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Discard invalid input
    //        cout << "Invalid input. Please enter an integer: ";
    //    }



    //    cout << "1:OMP 2:CUDA 3:MPI 4:Single Thread\nWhat parallel platform you want to use: ";
    //    while (!(cin >> method)) {  // Check if the input is of the correct type (int in this case)
    //        cin.clear();  // Clear the error flag on cin
    //        cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Discard invalid input
    //        cout << "Invalid input. Please enter an integer: ";
    //    }




    //    string imagePath;
    //    cin.ignore();  // Ignore leftover newline character
    //    cout << "Image Path: ";
    //    getline(cin, imagePath);

    //    Mat image = loadImage(imagePath);

    //    if (image.empty()) {
    //        cout << "Could not load the image!" << endl;
    //        return -1;
    //    }

    //    auto start = high_resolution_clock::now();

    //    switch (choice) {
    //    case 1:
    //        if (method == 1) {
    //        }
    //        else if (method == 2) {
    //        }
    //        else if (method == 3) {
    //            //processed_image = mpi_unsharp_masking(argc, argv);
    //            cout << "Please Run in PowerShell";
    //        }
    //        else if (method == 4) {
    //            processed_image = single_Laplacian_Filter(image);
    //        }


    //        break;
    //    case 2:
    //        if (method == 1) {
    //            processed_image = omp_unsharp_masking(image);
    //        }
    //        else if (method == 2) {
    //        }
    //        else if (method == 3) {
    //            //processed_image = mpi_Laplacian_Filter(argc, argv);
    //            cout << "Please Run in PowerShell";
    //        }
    //        else if (method == 4) {

    //            processed_image = single_thread_unsharp_masking(image);

    //        }

    //        break;
    //    }
    //    auto stop = high_resolution_clock::now();

    //    auto duration = duration_cast<milliseconds>(stop - start);

    //    

    //    // Display the original image
    //    namedWindow("Original Image", WINDOW_AUTOSIZE);
    //    imshow("Original Image", image);
    //    moveWindow("Original Image", 0, 0);

    //    // Load and display the processed image
    //    if (!processed_image.empty()) {
    //        namedWindow("Processed Image", WINDOW_AUTOSIZE);
    //        imshow("Processed Image", processed_image);
    //        moveWindow("Processed Image", 500, 0);
    //    }
    //    else {
    //        cout << "Could not load the processed image!" << endl;
    //    }

    //    waitKey(0); // Wait for a key press
    //    destroyAllWindows();

    //    cout << "Time taken for to precess: " << duration.count()/1000 << "seconds" << endl;
    //    char ans;
    //    cout << "Want to continue? (y/n)" << endl;
    //    cin >> ans;
    //    ans = tolower(ans);

    //    // Check if the input is valid (either 'y' or 'n')
    //    if (ans == 'y') {
    //       flag = true;
    //    }else{
    //        flag = false;
    //    }
    //    cout << flag << endl;


    //} while (flag);
    
    mpi_unsharp_masking(argc, argv);
    //mpi_Laplacian_Filter(argc, argv);
    return 0;

}