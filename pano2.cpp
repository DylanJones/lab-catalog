#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
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

Mat extractLabels(Mat img) {
    Mat grayscale, thresh;
    cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
    cv::threshold(grayscale, thresh, 210, 255, cv::THRESH_BINARY);
    //showImage("Thresh", thresh, 1);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::Mat(6, 6, CV_8UC1, 1));
    //showImage("Denoised", thresh, 1);
    vector<vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RetrievalModes::RETR_TREE,
                     cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
    vector<vector<cv::Point>> goodContours;
    // filter out wrong size ones
    for (auto cont : contours) {
        auto area = cv::contourArea(cont);
        if (area > 1000 && area < 5000) {
            double minX = 1e200, minY = 1e200, maxX = -1e200, maxY = -1e200;
            for (auto p : cont) {
#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))
                minX = MIN(minX, p.x);
                minY = MIN(minY, p.y);
                maxX = MAX(maxX, p.x);
                maxY = MAX(maxY, p.y);
            }
            double aspect = (maxX - minX) / (maxY - minY);
            if (aspect > 2 && aspect < 4) {
                goodContours.emplace_back(cont);
            }
        }
    }
    Mat mask = cv::Mat::zeros(cv::Size(grayscale.size[1], grayscale.size[0]), CV_8UC1);
    cv::drawContours(mask, goodContours, -1, cv::Scalar(255, 255, 255), -1);
    //cv::drawContours(img, goodContours, -1, cv::Scalar(255, 255, 255), 5);
    Mat labels;
    img.copyTo(labels, mask);
    //showImage("Labels", labels, 1);
    return labels;
}

int main(int argc, char **argv) {
    ////////////BAD//////////
    // Algorithm: 1) Find all the gray and their center coordinates.
    //            1a) (optional) filter out gray based on if they exist in adjacent frames
    //            2) Align the gray to a grid.  They should roughly fit based on current x,y.
    //            3) Construct the theoretical gray based on the rough grid.
    //            4) Find the homography matrix between the real and theoretical points and warpPerspective
    //               the image into a flat one.
    std::string filename("bell.mp4");
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    if (argc > 1) {
        filename = argv[1];
    }
    cv::VideoCapture video(filename); // file
    //cv::VideoCapture video(0); // camera

    Mat img, prevGray, gray;
    vector<cv::Point2f> points, prevPoints, referencePoints;

    // Initalization / first frame
    video.read(img);
    cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    //prevGray = extractLabels(img);
    //cv::resize(img, img, cv::Size(img.size[1] / 4, img.size[0] / 4));
    cv::cvtColor(img, prevGray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(prevGray, prevPoints, 500, 0.05, 10);
    cv::cornerSubPix(prevGray, prevPoints, cv::Size(10, 10), cv::Size(-1, -1), termcrit);
    referencePoints = prevPoints;
    Mat prevTransforms = Mat::eye(cv::Size(3, 3), CV_64F);

    Mat bigImg(img.size[1] * 10, img.size[0], CV_8UC3);

    while (video.read(img)) {
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
        //gray = extractLabels(img);
        //cv::resize(img, img, cv::Size(img.size[1] / 4, img.size[0] / 4));
        //showImage("Labels", gray, 0);
        // convert to grayscale for better detection
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, points, status, err);
        cv::cornerSubPix(gray, points, cv::Size(10, 10), cv::Size(-1, -1), termcrit);
        for (auto e : err)
            cout << int(e) << " ";
        cout << endl;

        for (auto pt : points) {
            circle(img, pt, 5, cv::Scalar(0, 255, 0), -1);
        }

        showImage("img", img, 1);

        // Based on relative points transform the image
        // select only good points
        vector<cv::Point2f> goodPoints, oldGoodPoints;
        for (int i = 0; i < points.size(); i++) {
            if (status[i]) {
                goodPoints.emplace_back(points[i]);
                oldGoodPoints.emplace_back(referencePoints[i]);
            }
        }
        Mat warped, mask;
        mask = Mat::ones(img.size[0], img.size[1], CV_8UC1);
        //Mat transform;
        //transform = cv::estimateAffine2D(goodPoints, oldGoodPoints);
        //cout << transform.size[1] << " " << transform.size[0] << endl;
        //cv::warpAffine(img, warped, prevTransforms * transform, cv::Size(img.size[1] * 10, img.size[0]));
        //cv::warpAffine(mask, mask, prevTransforms * transform, cv::Size(img.size[1] * 10, img.size[0]));
        Mat homography = cv::findHomography(goodPoints, oldGoodPoints);
        cout << homography.type() << endl;
        cv::warpPerspective(img, warped, prevTransforms * homography, cv::Size(img.size[1] * 10, img.size[0]));
        cv::warpPerspective(mask, mask, prevTransforms * homography, cv::Size(img.size[1] * 10, img.size[0]));
        cv::erode(mask, mask, Mat::ones(6, 6, CV_8UC1));

        Mat tmp;
        cv::Laplacian(gray, tmp, CV_64F);
        cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(tmp, mean, stddev, Mat());
        double variance = stddev.val[0] * stddev.val[0];
        if (variance > 235) {
            warped.copyTo(bigImg, mask);
        }
        showImage("warp", warped, 1);
        showImage("pano", bigImg, 1);

        gray.copyTo(prevGray);
        prevPoints = points;

        // Check if we need to reset
        int sum = 0;
        for (auto x : status)
            sum += x;
        if ((float) sum / status.size() < 0.9) { // We're not in kansas anymore
            cv::goodFeaturesToTrack(prevGray, prevPoints, 500, 0.05, 10);
            cv::cornerSubPix(prevGray, prevPoints, cv::Size(10, 10), cv::Size(-1, -1), termcrit);
            //cv::perspectiveTransform(prevPoints, prevPoints, homography);
            prevTransforms = prevTransforms * homography;
            referencePoints = prevPoints;
        }
    }
    cv::imwrite("pano.png", bigImg);
    return 0;
}
