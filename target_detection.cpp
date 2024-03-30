#include"target_detection.h"

std::pair<cv::Mat, int> target_detection(const cv::Mat& C_hat, double threshold, int max_margin, const cv::Mat& I_in) {
    int row = C_hat.rows;
    int col = C_hat.cols;
    cv::Mat mask = cv::Mat::zeros(cv::Size(row,col), CV_8UC1);

    int target_pixel_num = 0;
    for (int i = 0; i < C_hat.rows; i++) {
        for (int j = 0; j < C_hat.cols; j++) {
            if (C_hat.at<float>(i, j) > threshold) {
                //bug,i，j不知道为什么反了
                mask.at<uchar>(j, i) = 100;
                target_pixel_num++;
            }
        }
    }


    return std::pair<cv::Mat, int>(mask, target_pixel_num);
}