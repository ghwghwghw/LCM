% ��������"A Local Contrast Method for Small Infrared Target Detection"�е�Algorithm 2
% ����patch�ĳߴ������3�ı������������Ķ���u�Ĵ�Сֻ����3x3,5x5,7x7,9x9��patch�ĳߴ��Ӧ9x9,15x15,21x21,27x27
function [C_hat,max_margin]  = MLCM_computation(I_MLCM_in)
I_MLCM_in = double(I_MLCM_in);
[row,col] = size(I_MLCM_in);       
scales = [9,15,21,27];   % patch�ĳߴ���9x9,15x15,21x21,27x27
l_max = size(scales);    % ��Ӧ����lmax=[1,4],l_max(2)��4

% Compute Cl according to Algorithm 1
C_map_scales = zeros(row,col,l_max(2)); 
for i = 1:l_max(2)          % ��Ӧ��ͬ�߶�
    for j = 1:row-scales(i)+1  % ��һ�߶�����patchΪ��λ��������j����
        for k = 1:col-scales(i)+1                             %k����

            temp_patch = I_MLCM_in(j:j+scales(i)-1, k:k+scales(i)-1);%�ӣ�1��1�������Ͻ��Ǹ��㣩��ʼ��СΪscales(i)��scales(i)��Ϊ����
            %ԭ����ÿ�λ������ں�Ḳ����һ�ε�ֵ
            C_n  = LCM_computation(temp_patch);    % ��patchִ��Algorithm 1
            C_map_scales((2*j+scales(i)-1)/2, (2*k+scales(i)-1)/2,i) = C_n;
        end
    end
end
% �ⲿ�ּ��㣬����4�ŶԱȶ�ͼ�����г߶����ĶԱȶ�ͼ��Ч��Ԫ����С��ÿ�������ȥ(scales(4)-1)/2=13
max_margin = (scales(4)-1)/2;

% ���4�ֳ߶ȵĶԱ�ͼ
figure()
[X,Y] = meshgrid(1:1:row,1:1:col);
subplot(2,2,1); mesh(X,Y,C_map_scales(:,:,1)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=3x3 Contrast Map');
subplot(2,2,2); mesh(X,Y,C_map_scales(:,:,2)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=5x5 Contrast Map');
subplot(2,2,3); mesh(X,Y,C_map_scales(:,:,3)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=7x7 Contrast Map');
subplot(2,2,4); mesh(X,Y,C_map_scales(:,:,4)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=9x9 Contrast Map');

% ��4�ֳ߶ȶԱ�ͼ�Ĺ�ͬ����ȡ���ֵ����Ϊ���
C_hat = zeros(row-scales(4)+1,col-scales(4)+1);
for i = 1:row-scales(4)+1
    for j = 1:row-scales(4)+1
        temp = [C_map_scales(i+max_margin,j+max_margin,1);...
                C_map_scales(i+max_margin,j+max_margin,2);...
                C_map_scales(i+max_margin,j+max_margin,3);...
                C_map_scales(i+max_margin,j+max_margin,4)];
        C_hat(i,j) = max(temp);
    end
end

end
