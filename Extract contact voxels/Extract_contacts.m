clc;
clear;
%This script is used to extract contact voxels from CT images.
w_input='C:\Users\John cheng\Desktop\JoVE\Watershed lines\';
b_input='C:\Users\John cheng\Desktop\JoVE\Binary_2\';
c_output='C:\Users\John cheng\Desktop\JoVE\Contact voxels\';
%------------------------------input data from image--------------------
img_path_list = dir(strcat(w_input,'*.tif'));
img_num = length(img_path_list);%obtain the total number of image files 
imag_start=1;
imag_end=img_num;
if img_num > 0
            k=1;
        for j = imag_start:imag_end 
            image_name = img_path_list(j).name;% The name of the image
            image =  imread(strcat(w_input,image_name));
            w_lines(:,:,k)=image;   
            k=k+1;
        end
end

%------------------------------input data from image--------------------
img_path_list = dir(strcat(b_input,'*.tif'));
img_num = length(img_path_list);%obtain the total number of image files 
imag_start=1;
imag_end=img_num;
if img_num > 0
            k=1;
        for j = imag_start:imag_end 
            image_name = img_path_list(j).name;% The name of the image
            image =  imread(strcat(b_input,image_name));
            binary(:,:,k)=image;   
            k=k+1;
        end
end
%--------------------------------------------------------------------
Contacts=w_lines&binary;

%---------------------------output contacts--------------------------
for s=1:1:size(Contacts,3)
            imwrite( Contacts(:,:,s),[c_output , num2str(s,'%05d') '.tif']);
end

%--------------------------Typical results on slices-----------------
t=1000;
sign=['Slice ',num2str(t),'along z axis'];
struct=regionprops(logical(Contacts(:,:,t)),'PixelIdxList');
ori_D1=zeros(size(Contacts,1),size(Contacts,2));
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
 saveas(h,'Contacts of slice 1000','fig');