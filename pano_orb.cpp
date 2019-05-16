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
    //return;
    // make window
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    // show image
    Mat tmp;
    //cv::resize(image, tmp, cv::Size(image.size[1]/4, image.size[0]/4));
    //cv::imshow(winName, tmp);
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
    ////////////BAD//////////
    // Algorithm: 1) Find all the good features in the first frame, save as the "keyframe".
    //            2) Track the features to the next frame.  
    //            3) Find homography between the old points and the new ones, then transform the image to match the plane of the old
    //            4) If the amount of tracked points is small, reset the tracked points to ones found in this frame and add the current transformation to the accumulated one.
    std::string filename("betterbell.mp4");
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    if (argc > 1) {
        filename = argv[1];
    }
    cv::VideoCapture video(filename); // file
    //cv::VideoCapture video(0); // camera

    Mat img, prevImg, prevGray, gray, descriptors, prevDescriptors;
    auto featureDetector = cv::ORB::create();
    auto matcher = cv::BFMatcher::create(cv::NORM_L2SQR, true);
    vector<cv::KeyPoint> points, prevPoints, referencePoints;

    // Initalization / first frame
    video.read(prevImg);
    cv::rotate(prevImg, prevImg, cv::ROTATE_90_CLOCKWISE);
    cv::resize(prevImg, prevImg, cv::Size(prevImg.size[1] / 4, prevImg.size[0] / 4));
    cv::cvtColor(prevImg, prevGray, cv::COLOR_BGR2GRAY);
    featureDetector->detectAndCompute(prevImg, Mat(), prevPoints, prevDescriptors);

    Mat prevTransforms = Mat::eye(cv::Size(3, 3), CV_64F);
    Mat bigImg(prevImg.size[1] * 10, prevImg.size[0], CV_8UC3);

    int i = 0;
    while (video.read(img)) {
        cout << "Frame " << i++ << endl;
        if (i % 2 == 0) {
            continue;
        }
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
        cv::resize(img, img, cv::Size(img.size[1] / 4, img.size[0] / 4));
        // convert to grayscale for better detection
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        showImage("gray", gray, 1);

        vector<vector<cv::DMatch>> matches;
        Mat out;

        featureDetector->detectAndCompute(img, Mat(), points, descriptors);
        matcher->knnMatch(prevDescriptors, descriptors, matches, 1, cv::noArray(), true);
        cv::drawMatches(prevImg, prevPoints, img, points, matches, out);

        for (const auto &vec : matches) {
            const auto &match = vec[0];
            if (match.distance < 20) {

            }
        }

        showImage("img", out, 0);
while (1){;}
        // Based on relative points transform the image
        // Mat warped, mask;
        // mask = Mat::ones(img.size[0], img.size[1], CV_8UC1);
        // Mat homography = cv::findHomography(goodPoints, oldGoodPoints);
        // cv::warpPerspective(img, warped, prevTransforms * homography, cv::Size(img.size[1] * 10, img.size[0]));
        // cv::warpPerspective(mask, mask, prevTransforms * homography, cv::Size(img.size[1] * 10, img.size[0]));
        // cv::erode(mask, mask, Mat::ones(6, 6, CV_8UC1));

        // Mat tmp;
        // cv::Laplacian(gray, tmp, CV_64F);
        // cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
        // meanStdDev(tmp, mean, stddev, Mat());
        // double variance = stddev.val[0] * stddev.val[0];
        // if (variance > 235) {
        //     warped.copyTo(bigImg, mask);
        // }
        // showImage("warp", warped, 1);
        // showImage("pano", bigImg, 1);

        prevImg = img;
        prevGray = gray;
        prevPoints = points;
        prevDescriptors = descriptors;

        // Check if we need to reset
        //if ((float) sum / status.size() < 0.9) { // We're not in kansas anymore
            // prevTransforms = prevTransforms * homography;
        //}
    }
    cv::imwrite("pano.png", bigImg);
    return 0;
}
