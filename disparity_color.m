cutout_ratio = 8;
divider = 200;
f = 16/1000 / 3.75 * 1000000;
b = 600/1000;

img_num = 1248;
input_folder = 'input/';
output_folder = 'output/';
I = imread(strcat(output_folder,num2str(img_num),'d_planefitted.png'));
%I = imread(strcat(input_folder,num2str(img_num),'d.png'));

sz = size(I);
I_cutout = I(:,(sz(2)/cutout_ratio+1):sz(2));
I_corrected = I_cutout / divider;
I_color_disp = ind2rgb(I_corrected,jet(170));

imwrite(I_color_disp,strcat(output_folder,num2str(img_num),'d_planefitted_color.png'));
%imwrite(I_color_disp,strcat(output_folder,num2str(img_num),'d_color.png'));

figure(1), imshow(I_color_disp);
title("Click locations to get depth information!");

%% compute distance
for j = 1:5
    [x,y] = ginput;
    objDispVal = I_cutout(cast(y,'uint16'),cast(x,'uint16')) / divider;
    distVal = f*b/cast(mean(mean(objDispVal)),'double');
    disp(strcat("dist(m): ", num2str(distVal)));
end
