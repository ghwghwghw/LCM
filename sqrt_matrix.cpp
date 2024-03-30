#include"sqrt_matrix.h"

double sqrtMatrix(const cv::Mat& C_hat, double mean_C_hat) {
    // ת��Ϊ˫�����ͣ�����Ѿ�����Ӱ��
    C_hat.convertTo(C_hat, CV_32F);

    int row = C_hat.rows;
    int col = C_hat.cols;
    int Num = row * col;

    double sum = 0;

    // ���������е�ÿ��Ԫ��
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            double diff = C_hat.at<float>(i, j) - mean_C_hat;
            sum += diff * diff; // ������ƽ�������ۼӵ�sum��
        }
    }

    double sqrt_C_hat = sum / (Num - 1); // ������������

    return sqrt(sqrt_C_hat);
}