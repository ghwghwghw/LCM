#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


std::pair<cv::Mat, int> MLCM_computation(const cv::Mat& I_MLCM_in);