#include"lcm_computation.h"// 导入 Algorithm 1 的函数原型
#include"mlcm_computation.h" 
using namespace std;

std::pair<cv::Mat, int> MLCM_computation(const cv::Mat& I_MLCM_in) {
    cv::Mat I_MLCM_in_double;
    I_MLCM_in.convertTo(I_MLCM_in_double,CV_32F);

    int row = I_MLCM_in_double.rows;
    int col = I_MLCM_in_double.cols;
    double c_n;

    int scales[] = {9,15,21,27};
    int l_max = sizeof(scales) / sizeof(scales[0]);

    //创建C_map_scales
    std::vector<cv::Mat> C_map_scales(l_max);

    //Compute Cl according to Algorithm 1
    for (int i = 0; i < l_max; i++) {
        C_map_scales[i] = cv::Mat::zeros(row, col, CV_32F);
        
        //j是行
        for (int j = 0; j < row - scales[i] + 1; j++) {
            //k是列，先遍历列
            for (int k = 0; k < col - scales[i] + 1; k++) {
                //从（1，1）（左上角那个点）开始大小为scales(i)×scales(i)作为输入
                //先遍历列，再遍历行，大小为scales[i]
                cv::Mat temp_patch = I_MLCM_in_double(cv::Rect(j,k, scales[i], scales[i]));
                // 调用 Algorithm 1
                double c_n = LCM_computation(temp_patch);
                //生成contrast map
                C_map_scales[i].at<float>((2 * j + scales[i]) / 2, (2 * k + scales[i]) / 2) = c_n;

            }
        }
    }


    //max_margin
    int max_margin = (scales[l_max - 1] - 1) / 2;

    //对4种尺度对比图的共同部分取最大值，作为输出
    cv::Mat C_hat = cv::Mat::zeros(cv::Size(row-scales[3]+1, row - scales[3] + 1), CV_32FC1);

    std::vector<double> temp(l_max);
    for (int i = 0; i < row - scales[3] + 1; i++) {
        for (int j = 0; j < col - scales[3] + 1; j++) {
            // 清空 temp 向量
            temp.clear();

            //取一个像素点四通道中最大的
            for (int k = 0; k < l_max; k++) {
                double val = C_map_scales[k].at<float>(i + max_margin, j + max_margin);
                temp.push_back(val);
            }

            double max_val = *std::max_element(temp.begin(), temp.end());
            C_hat.at<float>(i, j) = max_val;
        }
    }

    return std::pair<cv::Mat, int>(C_hat, max_margin);
}
