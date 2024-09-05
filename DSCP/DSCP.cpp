#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string imagePath;

    cout << "Enter the path of the image: ";
    getline(cin, imagePath);

    // Load the image
    Mat image = imread("Image\\" + imagePath, IMREAD_COLOR);

    if (image.empty()) {
        cout << "Could not load the image!" << endl;
        return -1;
    }

    // Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    moveWindow("Original Image", 0, 0);

    waitKey(0); // Wait for a key press
    return 0;
}