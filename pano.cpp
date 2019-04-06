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

auto getDist(const cv::Point3f &a, const cv::Point3f &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
};

// THIS ONE WORKS!!!!!!
int main(int argc, char **argv) {
    std::string filename("bell.mp4");
    if (argc > 1) {
        filename = argv[1];
    }
    cv::VideoCapture video(filename); // file

    Mat undistorted, distorted, grayscale;

    while (video.read(undistorted)) {
        cv::resize(undistorted, undistorted, cv::Size(960, 540));
        // convert to grayscale for better detection
        cv::cvtColor(undistorted, grayscale, cv::COLOR_BGR2GRAY);
        // edge detect
        Mat edges, thresh, denoised;
        cv::Canny(undistorted, edges, 10, 200, 3);
        showImage("Canny", edges, 1);
        // cv::adaptiveThreshold(grayscale, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 15);
        cv::threshold(undistorted, thresh, 100, 255, cv::THRESH_BINARY_INV);
        showImage("Thresh", thresh, 1);
        // denoise
        // this doesn't work at all
        // cv::fastNlMeansDenoising(thresh, denoised, 9);
        // this doesn't help
        // cv::dilate(thresh, denoised, Mat(), Point(-1, -1), 2);
        // cv::erode(denoised, denoised, Mat(), Point(-1, -1), 2);
        // showImage("Denoised", denoised, 1);
        // find lines
        vector<cv::Vec2f> lines;
        cv::HoughLines(edges, lines, 1, CV_PI/180, 350, 0, 0 );
        // draw lines
        for(auto line : lines) {
            float rho = line[0], theta = line[1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            cv::line(undistorted, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
        }
        cout << lines.size() << " lines" << endl;
        showImage("Lines", undistorted, 0);
    }
    return 0;
}
