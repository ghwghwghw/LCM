#pragma once
#include <vector>
#include<opencv2/opencv.hpp>

std::pair<cv::Mat, int> target_detection(const cv::Mat& C_hat, double threshold, int max_margin, const cv::Mat& I_in);