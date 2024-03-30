% ��������"A Local Contrast Method for Small Infrared Target Detection"�е�Algorithm 1
% ����patch�ĳߴ������3�ı������������Ķ���u�Ĵ�Сֻ����3x3,5x5,7x7,9x9��patch�ĳߴ��Ӧ9x9,15x15,21x21,27x27
function  C_n  = LCM_computation(patch_LCM_in)

[row,col] = size(patch_LCM_in);       % ��patch���ԣ���=��
patch_LCM_in = double(patch_LCM_in);  % ����������
% ��Ϊ3x3��cells������patch�ĳߴ磬����3*3��cell
cell_size = row/3;
% ��������cell�����ֵ
L_n = max (max( patch_LCM_in( cell_size+1:cell_size*2, cell_size+1:cell_size*2 ) ) ); %ѡ��patch�������������ֵ
L_n_2 = L_n^2;
% �����ܱ�cell�ľ�ֵ,�ܱ߹�3^2-1��cell,������£�
% 1 2 3
% 4 0 5
% 6 7 8
m_1 = mean( mean( patch_LCM_in( 1:cell_size,                1:cell_size ) ));
m_2 = mean( mean( patch_LCM_in( 1:cell_size,                cell_size+1:cell_size*2 ) ));
m_3 = mean( mean( patch_LCM_in( 1:cell_size,                cell_size*2+1:cell_size*3 ) ));
m_4 = mean( mean( patch_LCM_in( cell_size+1:cell_size*2,    1:cell_size ) ));
m_5 = mean( mean( patch_LCM_in( cell_size+1:cell_size*2,    cell_size*2+1:cell_size*3 ) ));
m_6 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  1:cell_size ) ));
m_7 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  cell_size+1:cell_size*2 ) ));
m_8 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  cell_size*2+1:cell_size*3 ) ));
% ����C_n
m_cell = [L_n_2/m_1; L_n_2/m_2; L_n_2/m_3; L_n_2/m_4; L_n_2/m_5; L_n_2/m_6; L_n_2/m_7; L_n_2/m_8];
C_n = min(m_cell);
% Replace the value of the central pixel with the Cn


end
