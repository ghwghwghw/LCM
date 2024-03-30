#include"lcm_computation.h"


// ����һ������ԭ�ͣ�ʵ�� Algorithm 1
double LCM_computation(const cv::Mat& patch_LCM_in) {
    int row = patch_LCM_in.rows; // patch ������
    int col = patch_LCM_in.cols; // patch ������

    // �ı���������Ϊ double
    cv::Mat patch_LCM_in_double;
    patch_LCM_in.convertTo(patch_LCM_in_double, CV_32F);

    // ��Ϊ3x3��cells������ patch �ĳߴ磬����3*3��cell
    int cell_size = row / 3;

    // �������� cell �����ֵ
    //Rect(int x, int y, int width, int height);
    cv::Mat center_cell = patch_LCM_in_double(cv::Rect(cell_size, cell_size, cell_size, cell_size));
    double L_n;
    cv::minMaxLoc(center_cell, nullptr, &L_n);
    
    //���Դ�ӡ
    //std::cout<<L_n<<std::endl;

    double L_n_2 = L_n * L_n;

    // �����ܱ� cell �ľ�ֵ
    double m_1 = cv::mean(patch_LCM_in_double(cv::Rect(0, 0, cell_size, cell_size)))[0];
    double m_2 = cv::mean(patch_LCM_in_double(cv::Rect(0, cell_size, cell_size, cell_size)))[0];
    double m_3 = cv::mean(patch_LCM_in_double(cv::Rect(0, cell_size * 2, cell_size, cell_size)))[0];
    double m_4 = cv::mean(patch_LCM_in_double(cv::Rect(cell_size, 0, cell_size, cell_size)))[0];
    double m_5 = cv::mean(patch_LCM_in_double(cv::Rect(cell_size, cell_size * 2, cell_size, cell_size)))[0];
    double m_6 = cv::mean(patch_LCM_in_double(cv::Rect(cell_size * 2, 0, cell_size, cell_size)))[0];
    double m_7 = cv::mean(patch_LCM_in_double(cv::Rect(cell_size * 2, cell_size, cell_size, cell_size)))[0];
    double m_8 = cv::mean(patch_LCM_in_double(cv::Rect(cell_size * 2, cell_size * 2, cell_size, cell_size)))[0];

    // ���� C_n
    std::vector<double> m_cell = { L_n_2 / m_1, L_n_2 / m_2, L_n_2 / m_3, L_n_2 / m_4,
                                   L_n_2 / m_5, L_n_2 / m_6, L_n_2 / m_7, L_n_2 / m_8 };
    double C_n = *std::min_element(m_cell.begin(), m_cell.end());
    
    //���Դ�ӡ
    //std::cout << "C_n:"<<C_n << std::endl;
    
  

    return C_n;
}
