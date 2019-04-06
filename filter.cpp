#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/photo.hpp>
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

int main(int argc, char **argv) {
    std::string filename("bell.mp4");
    if (argc > 1) {
        filename = argv[1];
    }
    float threshold = 220;
    if (argc > 2) {
        threshold = strtof(argv[2], nullptr);
    }
    std::string outFile("filtered.mp4");
    if (argc > 3) {
        outFile = argv[3];
    }
    cv::VideoCapture video(filename); // file
    cv::VideoWriter out(outFile, video.get(cv::CAP_PROP_FOURCC), 24,
                        cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT)));

    Mat img, distorted;

    while (video.read(img)) {
        Mat grayscale;
        cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
        cv::Laplacian(grayscale, distorted, CV_64F);
        cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(distorted, mean, stddev, Mat());
        double variance = stddev.val[0] * stddev.val[0];
        if (variance > threshold) {
            out.write(img);
            cout << "Variance: " << variance << endl;
            showImage("original", img, 1);
        }
    }
    return 0;
}
