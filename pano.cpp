#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
// #include <opencv2/ccalib.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/aruco.hpp>
//#include <opencv2/photo.hpp>
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
    // Algorithm: 1) Find all the labels and their center coordinates.
    //            1a) (optional) filter out labels based on if they exist in adjacent frames
    //            2) Align the labels to a grid.  They should roughly fit based on current x,y.
    //            3) Construct the theoretical labels based on the rough grid.
    //            4) Find the homography matrix between the real and theoretical points and warpAffine
    //               the image into a flat one.
    std::string filename("filtered.avi");
    if (argc > 1) {
        filename = argv[1];
    }
    cv::VideoCapture video(filename); // file

    Mat img, distorted, grayscale;

    while (video.read(img)) {
        //cv::resize(img, img, cv::Size(960, 540));
        // convert to grayscale for better detection
        cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
        // edge detect
        Mat edges, thresh;
        cv::threshold(grayscale, thresh, 210, 255, cv::THRESH_BINARY);
        showImage("Thresh", thresh, 1);
        cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::Mat(6, 6, CV_8UC1, 1));
        showImage("Denoised", thresh, 1);
        vector<vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RetrievalModes::RETR_TREE,
                         cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
        vector<vector<cv::Point>> goodContours;
        // filter out wrong size ones
        for (auto cont : contours) {
            auto area = cv::contourArea(cont);
            if (area > 1000 && area < 3000) {
                double minX = 1e200, minY = 1e200, maxX = -1e200, maxY = -1e200;
                for (auto p : cont) {
#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))
                    minX = MIN(minX, p.x);
                    minY = MIN(minY, p.y);
                    maxX = MAX(maxX, p.x);
                    maxY = MAX(maxY, p.y);
                }
                double aspect = (maxY - minY) / (maxX - minX);
                if (aspect > 2 && aspect < 4) {
                    goodContours.emplace_back(cont);
                }
            }
        }
        Mat mask = cv::Mat::zeros(cv::Size(grayscale.size[1], grayscale.size[0]), CV_8UC1);
        cv::drawContours(mask, goodContours, -1, cv::Scalar(255, 255, 255), -1);
        cv::drawContours(img, goodContours, -1, cv::Scalar(255, 255, 255), 5);
        Mat labels;
        img.copyTo(labels, mask);
        showImage("Labels", labels, 1);
        Mat points = Mat::zeros(cv::Size(labels.size[1], labels.size[0]), CV_8UC3);
        for (auto cont : goodContours) {
            auto moments = cv::moments(cont, false);
            cv::circle(points, cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00), 10,
                       cv::Scalar(0, 0, 255), 10);
        }
        showImage("Points", points, 0);
        //cout << "All: " << contours.size() << "Good: " << goodContours.size() << endl;

    }
    return 0;
}
