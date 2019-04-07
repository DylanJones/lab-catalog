#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <set>
#include <unordered_map>
#include <cstdlib>

using cv::Mat;
using cv::Point;
using std::vector;
using std::cout;
using std::endl;

void showImage(const std::string &winName, const Mat &image, const int delayMs = 0) {
    // make window
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    // show image
    cv::imshow(winName, image);
    if (delayMs == 0) {
        // pause until spacebar pressed
        int key;
        do {
            key = cv::waitKey(0);
            cout << "Key pressed: " << key << endl;
            if (key == 0x71) { // q
                exit(0);
            }
        } while (key != 0x20); // space
    } else {
        // only wait for the specified amount of time
        int key = cv::waitKey(delayMs);
        if (key == 0x71) { // q
            exit(0);
        }
    }
}

// THIS ONE WORKS!!!!!!
int main(int argc, char **argv) {
    std::string filename("bell.mp4");
    if (argc > 1) {
        filename = argv[1];
    }
    cv::VideoCapture video(filename); // file
    auto stitcher = cv::Stitcher::create(cv::Stitcher::Mode::SCANS);
    Mat img, pano;
    vector<Mat> images;

    while (video.read(img)) {
    //for (int i = 0; i < 80; i++) {
        video.read(img);
        images.emplace_back(img);
        showImage("img", img, 3);
    }

    auto retval = stitcher->stitch(images, pano);
    cout << retval << endl;
    showImage("pano", pano, 0);

    return 0;
}
