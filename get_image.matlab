%ctrl+r注释多行，ctrl+t把多行取消注释
%strcat(a,b,c,...)把多个字符串拼接起来
%mat_name = pic_name(1:end-4)把文件的后缀去掉仅提取文件名
%double(a)把a转变为double类
%zeros(), ones()定义固定大小的矩阵，输入必须为标量
%s_[1,:,:]=s, 对s_的第一维赋值
%type_s = unique(s)找出s中不同的元素，并存入type_s
%u = intersect(ans, type_0)找出两个矩阵中相同的元素
%label2rgb把标签图变为rgb图
%matlab中image画出来的图的data cursor的x和y分别对应列号和行号，特别反人类和size的结果是相反的
d = dir('G:\buildings');
d_pic ='G:\buildings';
d_seg ='F:\LMSun\LabelsSemantic';
target_d = 'D:\h5Data\data\coco\images';
%d_pic_map = 'D:\h5Data\data\coco\images\temp\temp.bmp';
path_file = 'D:\h5Data\data\coco\images\path_name\';
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
%f_mat = {}; %存储文件夹名字的矩阵
nameFolds = strcat(nameFolds);
%for i=1:length(nameFolds)
%     t = nameFolds(i);
%     t = char(t);
%     m = strcat(d_str,'\',t);
%     f_mat(i) = m;
% end
% f,length(nameFolds)
counter = 1;
q=0;
type_0 = [4 14 16 17 21 22 25 26 27 28 29 37 38 39 46 51 52 54 55 56 58 59 60 64 65 67 69 70 71 72 76 84 87 89 90 91 92 94 95 97 98 100 104 105 115 116 117 120 124 125 126 129 130 131 134 137 138 140 142 143 144 145 148 150 152 153 154 155 157 158 160 162 165 166 167 174 176 178 179 186 190 191 192 196 198 199 202 203 205 206 208 214 216 219 221 223 224 226 228];
type_1 = [1 32 50 83 88 127 146 159 161 185 210 211 215];
type_2 = [2 18 34 40 62 107 111 123 132 135 200 201 213 218 227 230];
type_3 = [3 6 9 10 15 19 20 36 42 61 68 74 78 79 80 81 82 106 109 110 113 122 133 139 141 149 151 163 180 181 193 207 232 ];
type_4 = [5 7 8 11 12 13 23 30 31 33 43 47 48 57 73 75 85 99 102 108 114 119 164 172 175 182 183 188 194 195 197 204 209 222 229 231 ];
type_5 = [35 41 44 121 156 212 220];
type_7 = [63 66 77 93 103 128 147 170 171 184 ];
type_8 = [53 187 ];
type_9 = [24];
type_10 = [96 118 169 177 225 ];
type_11 = [45 86 101 173 189 ];
type_12 = [49 112 136 168 217];
for i=1:length(nameFolds)
    t = nameFolds(i);
    t = char(t);
    f = strcat(d_pic, '\' ,t);
    dirOut = dir(fullfile(f,'*.jpg'));
    pic_list = {dirOut.name}';
    for j = 1:length(pic_list)
        pic_name = pic_list(j);
        pic_name = char(pic_name);
        outpath_pic = strcat(target_d,'\train\');
        outpath_pic_seg = strcat(target_d, '\train_seg\');
        outname_pic = ['in',num2str(counter),'.png'];
        outname_seg = ['in',num2str(counter),'.png'];
        mat_name = pic_name(1:end-4);
        mat_path =[d_seg,'\',t,'\',mat_name,'.mat'];
        mat_load = importdata(mat_path);
        pic_seg_label = label2idx(mat_load.S);
        pic_seg = zeros(size(mat_load.S));
        for j1 = 1:length(pic_seg_label);
            if(~isempty(pic_seg_label{j1}))
                pic_seg(pic_seg_label{j1})= j1;
            end
        end
        %figure;imshow(uint8(pic_seg))
        pic_seg = double(pic_seg);
        pic_seg_type = unique(pic_seg);
        pic_seg_type_1 = unique(mat_load.S);
        pic_seg_type0=intersect(pic_seg_type,type_0);
        pic_seg_type1=intersect(pic_seg_type,type_1);
        pic_seg_type2=intersect(pic_seg_type,type_2);
        pic_seg_type3=intersect(pic_seg_type,type_3);
        pic_seg_type4=intersect(pic_seg_type,type_4);
        pic_seg_type5=intersect(pic_seg_type,type_5);
        pic_seg_type7=intersect(pic_seg_type,type_7);
        pic_seg_type8=intersect(pic_seg_type,type_8);
        pic_seg_type9=intersect(pic_seg_type,type_9);
        pic_seg_type10=intersect(pic_seg_type,type_10);
        pic_seg_type11=intersect(pic_seg_type,type_11);
        pic_seg_type12=intersect(pic_seg_type,type_12);
        if(~isempty(pic_seg_type0))
            for k0 =1:length(pic_seg_type0);
                idx = find(pic_seg==pic_seg_type0(k0));
                pic_seg(idx) = 0;
            end
        end
        
        if(~isempty(pic_seg_type1))
            for k1 =1:length(pic_seg_type1);
                idx = find(pic_seg==pic_seg_type1(k1));
                pic_seg(idx) = 1.5;
            end
        end
        
        if(~isempty(pic_seg_type2))
            for k2 =1:length(pic_seg_type2);
                idx = find(pic_seg==pic_seg_type2(k2));
                pic_seg(idx) = 2.5;
            end
        end
        
        if(~isempty(pic_seg_type3))
            for k3 =1:length(pic_seg_type3);
                idx = find(pic_seg==pic_seg_type3(k3));
                pic_seg(idx) = 3.5;
            end
        end
        
        if(~isempty(pic_seg_type4))
            for k4 =1:length(pic_seg_type4);
                idx = pic_seg==pic_seg_type4(k4);
                pic_seg(idx) = 4.5;
            end
        end
        
        if(~isempty(pic_seg_type5))
            for k5 =1:length(pic_seg_type5);
                idx = find(pic_seg==pic_seg_type5(k5));
                pic_seg(idx) = 5.5;
            end
        end
        
        if(~isempty(pic_seg_type7))
            for k6 =1:length(pic_seg_type7);
                idx = find(pic_seg==pic_seg_type7(k6));
                pic_seg(idx) = 6.5;
            end
        end
        
        if(~isempty(pic_seg_type8))
            for k7 =1:length(pic_seg_type8);
                idx = find(pic_seg==pic_seg_type8(k7));
                pic_seg(idx) = 7.5;
            end
        end
        
        if(~isempty(pic_seg_type9))
            for k8 =1:length(pic_seg_type9);
                idx = find(pic_seg==pic_seg_type9(k8));
                pic_seg(idx) = 8.5;
            end
        end
        
        if(~isempty(pic_seg_type10))
            for k9 =1:length(pic_seg_type10);
                idx = find(pic_seg==pic_seg_type10(k9));
                pic_seg(idx) = 9.5;
            end
        end
        
        if(~isempty(pic_seg_type11))
            for k10 =1:length(pic_seg_type11);
                idx = find(pic_seg==pic_seg_type11(k10));
                pic_seg(idx) =10.5;
            end
        end
        
        if(~isempty(pic_seg_type12))
            for k11 =1:length(pic_seg_type12);
                idx = find(pic_seg==pic_seg_type12(k11));
                pic_seg(idx) = 11.5;
            end
        end
        [r,c]=size(pic_seg);
        pic_seg_save = zeros(r,c,3);
        r_ = [0 0 1 1 0 1 0.5 0 1 0.5 0.5 1];
        g_ = [0 1 1 0 0 1 0.5 1 0 1   0   0.5];
        b_ = [0 0 1 0 1 0 0.5 1 1 0   1   0];
        color_ = [r_;g_;b_];
        for k12 = 1:3
            pic_seg_temp = pic_seg;
            for k13 = 1.5:11.5
                idx = find(pic_seg_temp==k13);  %对type1~12进行处理，type0不管
                if(~isempty(idx))
                    pic_seg_temp(idx) = color_(k12,k13+0.5);
                end
            end
%             figure;image(pic_seg_temp)
             pic_seg_save(:,:,k12) = pic_seg_temp;
%             figure;image(pic_seg_save)
        end
        %pic_seg_save = uint8(pic_seg_save);
        imwrite(pic_seg_save,[outpath_pic_seg,outname_seg]);
        copyfile([f,'\',pic_name],[outpath_pic, outname_pic]);
        white_file = zeros(3,3);
        imwrite(white_file, [path_file,'in',num2str(counter),'__',t,'__',mat_name,'_mat','.jpg']);
        counter = counter +1;
        if(rem(counter,100)==0)
            fprintf('%d00\n',q)
            q=q+1;
        end
    end
end
