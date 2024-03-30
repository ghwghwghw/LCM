#include"sqrt_matrix.h"

double sqrtMatrix(const cv::Mat& C_hat, double mean_C_hat) {
    // 转换为双精度型，如果已经是则不影响
    C_hat.convertTo(C_hat, CV_32F);

    int row = C_hat.rows;
    int col = C_hat.cols;
    int Num = row * col;

    double sum = 0;

    // 遍历矩阵中的每个元素
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            double diff = C_hat.at<float>(i, j) - mean_C_hat;
            sum += diff * diff; // 计算差的平方，并累加到sum中
        }
    }

    double sqrt_C_hat = sum / (Num - 1); // 计算样本方差

    return sqrt(sqrt_C_hat);
}