#include<iostream>
#include<opencv2/opencv.hpp>
#include"lcm_computation.h"
#include"mlcm_computation.h" 
#include"sqrt_matrix.h"
#include"target_detection.h"

using namespace cv;
using namespace std;

int main() {
	//��ȡ�Ҷ�ͼ
	cv::Mat I_in = cv::imread("C:/Users/ghw/desktop/2.bmp",cv::ImreadModes::IMREAD_GRAYSCALE);
	cv::resize(I_in, I_in, cv::Size(256, 256));
	cv::imshow("����", I_in);
	//cout <<"width:" << I_in.cols << " height:" << I_in.rows <<" channels: " << I_in.channels() << endl;
	//תΪfloat
	I_in.convertTo(I_in, CV_32F);
    

    // ����һ������� 9x9 �Ҷ�ͼ��
    //cv::Mat I_in(29, 29, CV_8UC1);
    //cv::randu(I_in, 0, 10);
	//��ӡ���
    //std::cout <<"I_in" << std::endl << I_in << std::endl;

	//�õ�c_hat��max_margin
	std::pair<cv::Mat, int> result_MLCM = MLCM_computation(I_in);
	cv::Mat C_hat = result_MLCM.first;
	int max_margin = result_MLCM.second;

	//��ӡ���
	//std::cout << "C_hat:" << std::endl << C_hat << std::endl;
	

	//�����ֵ
	cv::Scalar mean_c_hat = cv::mean(C_hat);

	//��ӡ���
	//std::cout << "mean_val:" << std::endl << mean_c_hat[0] << std::endl;

	//�����׼��
	double sqrt_C_hat = sqrtMatrix(C_hat, mean_c_hat[0]);

	//��ӡ���
	//std::cout << "sqrt_C_hat:" << std::endl << sqrt_C_hat << std::endl;

	//������ֵ
	int k_Th = 4;
	double threshold = mean_c_hat[0] + k_Th * sqrt_C_hat;
	//��ӡ���
	//std::cout << "threshold:" << std::endl << threshold << std::endl;

	//������ֵ�жϣ������ֵ̽������ͳ��СĿ����mask��ռ�ݵ���Ԫ��

	std::pair<cv::Mat, int> result_out = target_detection(C_hat,threshold,max_margin,I_in);
	cv::Mat I_out = result_out.first;
	int target_pixel_num = result_out.second;

	//��ӡ���
	//std::cout << "I_out:" << std::endl << I_out << std::endl;
	//std::cout << "target_pixel_num:" << std::endl << target_pixel_num << std::endl;


	cv::imshow("��ֵ�����", I_out);
	cv::waitKey(0);

	return 0;
}