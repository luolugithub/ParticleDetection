clc;
clear;
file_path='C:\Users\John cheng\Desktop\JoVE\Binary_2\';
label_path='C:\Users\John cheng\Desktop\JoVE\Labelled\';
watershed_path='C:\Users\John cheng\Desktop\JoVE\Watershed lines\';
img_path_list = dir(strcat(file_path,'*.tif'));
img_num = length(img_path_list);%obtain the total number of image files 
imag_start=1;
imag_end=img_num;
if img_num > 0
            k=1;
        for j = imag_start:imag_end 
            image_name = img_path_list(j).name;% The name of the image
            image =  imread(strcat(file_path,image_name));
            A(:,:,k)=image;   
            k=k+1;
        end
end

B=logical(A);
clear A;
fraction=0.2;
[ lines, regions ] = m_watershed( B, fraction );% maker-based watershed segmentation
%%%%%Output watershed lines
for s=1:1:size(lines,3)
            imwrite( lines(:,:,s),[watershed_path , num2str(s,'%05d') '.tif']);
end
L=bwlabeln(regions);
%%%%%Output labelled image
for s=1:1:size(L,3)
            imwrite( L(:,:,s),[label_path , num2str(s,'%05d') '.tif']);
end
%%%%%Output segmentation results of a slice
t=1000;
sign=['Slice ',num2str(t),'along z axis'];
struct=regionprops(logical(L(:,:,t)),'PixelIdxList');
ori_D1=zeros(size(L,1),size(L,2));
for i=1:1:size(struct,1)
   s=randperm(20);
   ori_D1(struct(i).PixelIdxList)=s(1); 
end

map=[0 0 0
     1 1 0.5
     1 1 0
     1 0.5 1
     1 0 1
     0.5 1 1
     0 1 1
     0 0 0.5
     0 0 1
     0 1 0
     0 0.5 0
     1 0 0
     0.5 0 0
     0.5 0.5 1
     0.5 0.5 0
     0.5 0.5 0.5
     0.5 1 0.5
     0.5 0 0.5
     1 0.5 0.5
     0 0.5 0.5];
 h=figure (1); imshow(uint8(ori_D1),map); title(sign);
 saveas(h,'Labelled image of slice 1000','fig');
 
