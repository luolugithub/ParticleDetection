function [ lines, regions ] = m_watershed( b_image, fraction )
%UNTITLED Summary of this function goes here
%   b_image-----the binary image to be segmented
%   fraction-----the marker coefficient (0~1); increase of fraction will decrease the number of separated particles
%   lines-----the watershed lines
%   regions----the seperated regions
dist=bwdist(~b_image);
progress=1
if length(size(dist))==3
marker_level=double(max(max(max(dist)))).*fraction;
elseif length(size(dist))==2
marker_level=double(max(max(dist))).*fraction;
end
Merge_dist=imhmax(dist,marker_level);
clear dist;
invert_m=-Merge_dist;
% invert_m(find(invert_m==0))=Inf;
clear Merge_dist;
%-------------------------------for insufficient memorgy (for the number of slices is larger than 2, 000)-------------------------
t1=fix(size(b_image,3)./5);
t2=t1*2;
t3=t1*3;
t4=t1*4;
t5=size(b_image,3);
W1=watershed(invert_m(:,:,1:t1));
W(:,:,1:t1)=W1;
clear W1;
progress=2
W2=watershed(invert_m(:,:,t1+1:t2));
W(:,:,t1+1:t2)=W2;
progress=3
clear W2;
W3=watershed(invert_m(:,:,t2+1:t3));
W(:,:,t2+1:t3)=W3;
progress=4
clear W3;
W4=watershed(invert_m(:,:,t3+1:t4));
W(:,:,t3+1:t4)=W4;
progress=5
clear W4;
W5=watershed(invert_m(:,:,t4+1:t5));
W(:,:,t4+1:t5)=W5;
progress=5
clear W5;
%------correction
W12=watershed(invert_m(:,:,t1-200:t1+200));
W(:,:,t1-50:t1+50)=W12(:,:,151:251);
progress=5
clear W12;
W23=watershed(invert_m(:,:,t2-200:t2+200));
W(:,:,t2-50:t2+50)=W23(:,:,151:251);
clear W23;
W34=watershed(invert_m(:,:,t3-200:t3+200));
W(:,:,t3-50:t3+50)=W34(:,:,151:251);
clear W34;
W45=watershed(invert_m(:,:,t4-200:t4+200));
W(:,:,t4-50:t4+50)=W45(:,:,151:251);
clear W45;
%-------------------------------------------------------------------
% W=watershed(invert_m);
clear invert_m;
lines=W==0;
regions=b_image&~lines;
end

