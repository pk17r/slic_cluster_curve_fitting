function slic_stand_alone_fn(img_num)

% Implementation of Simple Linear Iterative Clustering (SLIC)
%
% Input:
%   - img: input color image
%   - K:   number of clusters
%   - compactness: the weights for compactness
% Output: 
%   - cIndMap: a map of type uint16 storing the cluster memberships
%     cIndMap(i, j) = k => Pixel (i,j) belongs to k-th cluster
%   - time:    the time required for the computation
%   - imgVis:  the input image overlaid with the segmentation

% Put your SLIC implementation here
% 1248, 1251, 1255, 1260, 1264, 1266, 1268, 1370, 1373, 1377, 1380, 1383, 1390, 1420, 1425, 
% 1428, 1430, 1488, 1491, 1493, 1496, 1498, 1501, 1503, 1506, 1508, 1510, 1512
% img_num = 1428
display(img_num)
inp_img_fold = '/mnt/win/WORK/kentland19jul/22m_extracted_data/left_rect/';
inp_disp_fold = '/mnt/win/WORK/kentland19jul/22m_extracted_data/disparities/';
output_folder = 'output/';
output_folder2 = '/home/pkr/pkr-work/pose_estimation/build/segmentlabels/';
if exist(strcat(output_folder2,num2str(img_num),'.png'), 'file')
    display('segmented map already exists');
    return;
end
if ~exist(strcat(inp_disp_fold,num2str(img_num),'.png'), 'file')
    display('disparity image does not exist');
    return;
end
disp_img = imread(strcat(inp_disp_fold,num2str(img_num),'.png'));
disp_img_cutout = disp_img(101:620,161:1180);
disp_img_cutout = cast(disp_img_cutout,'double');
disp_img_cutout = disp_img_cutout(:);
disp_img_cutout_var = var(disp_img_cutout)
if disp_img_cutout_var > 10
    display('var in disparity > 10');
    return;
end
clear disp_img_cutout_var disp_img_cutout disp_img
if ~exist(strcat(inp_img_fold,num2str(img_num),'.png'), 'file')
    display('left image does not exist');
    return;
end

img = imread(strcat(inp_img_fold,num2str(img_num),'.png'));
K = 1024;
compactness = 20;   %40
E_threshold = 50;

tic;
% Input data
% imgB   = im2double(img);
imgB   = cast(img, 'double');
cform  = makecform('srgb2lab');
imgLab = applycform(im2double(img), cform);
%imgLab = imgB;
% Initialize cluster centers (equally distribute on the image). Each cluster is represented by 5D feature (L, a, b, x, y)
% Hint: use linspace, meshgrid
[s_vert, s_hori, ~] = size(imgLab); % get the size of image
%s = sqrt(s_hori*s_vert/K);  % grid steps s
nx = sqrt(K);
ny = nx;
sx = s_hori/nx;
sy = s_vert/ny;

iniX = linspace(sx/2,s_hori-sx/2,nx); % get the evenly distributed x values
iniY = linspace(sy/2,s_vert-sy/2,ny); % get the evenly distributed y values
C = zeros(K,5);

% figure(1), imshow(img), hold on;
% for i=1:length(iniX)
%     for j=1:length(iniY)
%         r = rectangle('Position',[iniX(i)-sx/2, iniY(j)-sy/2 sx sy]');
%         r.EdgeColor = 'b';
%         r.LineWidth = 3;
%         hold on;
%     end
% end
% hold off;

k = 1;
for i=1:length(iniX)
    for j=1:length(iniY)
        x = iniX(i);
        y = iniY(j);
        l = interp2(imgLab(:,:,1),x,y,'bilinear');
        a = interp2(imgLab(:,:,2),x,y,'bilinear');
        b = interp2(imgLab(:,:,3),x,y,'bilinear');
        C(k,:) = [l,a,b,x,y];
        k = k + 1;
    end
end

% move the cluster centers to lowest gradient points in 3x3 space
C2 = zeros(K,5);
[Ix, Iy] = imgradientxy(rgb2gray(img));
[X,Y] = meshgrid(-1:1,-1:1);
for k = 1:K
    x = C(k,4);
    y = C(k,5);
    Xp = X + x;
    Yp = Y + y;
    Px = interp2(Ix,Xp,Yp,'bilinear');
    Py = interp2(Iy,Xp,Yp,'bilinear');
    Pxy2 = Px^2 + Py^2;
    [M,Ind] = min(Pxy2(:));
    [row, col] = ind2sub(size(Pxy2),Ind);
    x_min = Xp(col);
    y_min = Yp(row);
    l = interp2(imgLab(:,:,1),x_min,y_min,'bilinear');
    a = interp2(imgLab(:,:,2),x_min,y_min,'bilinear');
    b = interp2(imgLab(:,:,3),x_min,y_min,'bilinear');
    C2(k,:) = [l,a,b,x_min,y_min];
end

C = C2;

% SLIC superpixel segmentation
% In each iteration, we update the cluster assignment and then update the cluster center
label_map = -ones(s_hori,s_vert);
dist_map = Inf(s_hori,s_vert);
m = compactness;
numIter  = 10; % Number of iteration for running SLIC
e = 0;
% figure(2), imshow(img), hold on;
for iter = 1: numIter
    for k = 1:K
        % 1) Update the pixel to cluster assignment
        lk = C(k,1);
        ak = C(k,2);
        bk = C(k,3);
        xk = C(k,4);
        yk = C(k,5);
        for i = -ceil(sx)-1:1:ceil(sx)+1
            for j = -ceil(sy)-1:1:ceil(sy)+1
                xp = round(xk) + i;
                yp = round(yk) + j;
                if or(xp < 1, xp > s_hori)
                    continue;
                end
                if or(yp < 1, yp > s_vert)
                    continue;
                end
                if round(xp) - xp ~= 0
                    print('bug');
                end
                if round(yp) - yp ~= 0
                    print('bug');
                end
                lp = imgLab(yp,xp,1);
                ap = imgLab(yp,xp,2);
                bp = imgLab(yp,xp,3);
                dc2 = (lp - lk)^2 + (ap - ak)^2 + (bp - bk)^2;
                ds2 = (xp - xk)^2 + (yp - yk)^2;
                D = sqrt(dc2 + ds2 * m^2 / sx / sy);
                if D < dist_map(xp,yp)
                    dist_map(xp,yp) = D;
                    label_map(xp,yp) = k;
                end
            end
        end
    end
    zeroVec = find(label_map==-1);
    if length(zeroVec) ~= 0
        print('bug');
    end
    
	% 2) Update the cluster center by computing the mean
    e2 = e;
    e = 0;
	for k = 1:K
        xk = C(k,4);
        yk = C(k,5);
        [Xc,Yc] = find(label_map==k);
%         plot(Xc, Yc, 'color', rand(1,3));
        X_mean = mean(Xc);
        Y_mean = mean(Yc);
        l_mean = 0;
        a_mean = 0;
        b_mean = 0;
        for i = 1:length(Xc);
            if or(Xc(i) > s_hori, Yc(i) > s_vert)
                print('bug');
            end
            l_mean = l_mean + mean(imgLab(Yc(i),Xc(i),1));
            a_mean = a_mean + mean(imgLab(Yc(i),Xc(i),2));
            b_mean = b_mean + mean(imgLab(Yc(i),Xc(i),3));
        end
        l_mean = round(l_mean / length(Xc));
        a_mean = round(a_mean / length(Xc));
        b_mean = round(b_mean / length(Xc));
        C(k,:) = [l_mean,a_mean,b_mean,X_mean,Y_mean];
        e = e + sqrt((X_mean - xk)^2 + (Y_mean - yk)^2);
    end
    
%     %visualize:
    cIndMap1 = label_map';
    [gx, gy] = gradient(cIndMap1);
    bMap = (gx.^2 + gy.^2) > 0;
    imgVis = img;
    imgVis(cat(3, bMap, bMap, bMap)) = 1;
%     figure(2), imshow(imgVis)
    display(strcat('pass: ',num2str(iter),', e: ',num2str(e)));
    
    if or(e2 == e, e < E_threshold)
        break;
    end
end

% figure(2), imshow(imgVis);
% title(num2str(img_num));
imwrite(imgVis,strcat(output_folder,num2str(img_num),'_segmented.png'));

%% 3) Combine similar clusters

cluster_D_threshold = 20; % 14.6
C2 = zeros(K,6);
C2(:,1:3) = C(:,1:3);

variance_multiplier = 2;
% calculate std. div. in a cluster
for i=1:K
    [Yc,Xc] = find(cIndMap1 == i);
    l = zeros(length(Xc),1);
    a = l;
    b = l;
    for j=1:length(Xc)
        l(j) = imgLab(Yc(j),Xc(j),1);
        a(j) = imgLab(Yc(j),Xc(j),2);
        b(j) = imgLab(Yc(j),Xc(j),3);
    end
    lstd = std(l);
    astd = std(a);
    bstd = std(b);
    C2(i,4) = lstd * variance_multiplier;
    C2(i,5) = astd * variance_multiplier;
    C2(i,6) = bstd * variance_multiplier;
end

K2 = K;
clusters_combined = 0;
C_parent = [1:K2];
C3 = zeros(K2,6);
label_map2 = label_map;
new_cluster_index = 1;
[Idx, dis] = knnsearch(C2,C2,'Distance','euclidean','K',K2);
for i=1:K2
    if C_parent(i) ~= i
        % don't look at clusters which have already been combined
        continue;
    end
    nearby_clusters = Idx(i, find(dis(i,:) < cluster_D_threshold));
    if length(nearby_clusters) == 1 && nearby_clusters == i
        % no match found
        C3(new_cluster_index,:) = C2(nearby_clusters,:);
        new_cluster_index = new_cluster_index + 1;
        continue;
    end
    C_parent(nearby_clusters) = new_cluster_index;
    C3(new_cluster_index,1) = mean(C2(nearby_clusters,1));
    C3(new_cluster_index,2) = mean(C2(nearby_clusters,2));
    C3(new_cluster_index,3) = mean(C2(nearby_clusters,3));
    l2 = [];
    a2 = [];
    b2 = [];
    for j = 1:length(nearby_clusters)
        [Xc,Yc] = find(label_map2 == nearby_clusters(j));
        l = zeros(length(Xc),1);
        a = l;
        b = l;
        for p=1:length(Xc)
            label_map2(Xc(p),Yc(p)) = new_cluster_index;
            l(p) = imgLab(Yc(p),Xc(p),1);
            a(p) = imgLab(Yc(p),Xc(p),2);
            b(p) = imgLab(Yc(p),Xc(p),3);
        end
        l2 = [l2; l];
        a2 = [a2; a];
        b2 = [b2; b];
    end
    lstd = std(l2);
    astd = std(a2);
    bstd = std(b2);
    C3(new_cluster_index,4) = lstd * variance_multiplier;
    C3(new_cluster_index,5) = astd * variance_multiplier;
    C3(new_cluster_index,6) = bstd * variance_multiplier;
    clusters_combined = clusters_combined + length(nearby_clusters) - 1;
    new_cluster_index = new_cluster_index + 1;
end
K2 = new_cluster_index - 1;
%C2 = C3(1:new_cluster_index-1,:);
%C4 = zeros(K2,3);
C_parent = [1:K2];

% renumbering the cluster labels and removing noise
P = 2;
size_label_map2 = size(label_map2);
label_map_no_noise = zeros(size_label_map2(1),size_label_map2(2));
new_cluster_index = 1;
for i=1:K
    [Xc,Yc] = find(label_map2 == i);
    BW = false(size_label_map2(1),size_label_map2(2));
    for p=1:length(Xc)
        BW(Xc(p),Yc(p)) = true;
    end
    BW2 = bwareaopen(BW,P);
    [Xc,Yc] = find(BW2 == true);
    if length(Xc) > 0
        for p=1:length(Xc)
            label_map_no_noise(Xc(p),Yc(p)) = new_cluster_index;
        end
        new_cluster_index = new_cluster_index + 1;
    end
    continue;
end

% save
cIndMap = label_map_no_noise';
[gx, gy] = gradient(cIndMap);
bMap = (gx.^2 + gy.^2) > 0;
imgVis = img;
imgVis(cat(3, bMap, bMap, bMap)) = 1;
%figure(1), imshow(img)
figure(3), imshow(imgVis)
title(num2str(img_num));

imwrite(imgVis,strcat(output_folder,num2str(img_num),'_segments_joined.png'));

% imwrite(cast(label_map_no_noise','uint8'),strcat(output_folder,num2str(img_num),'_segment_map.png'));
imwrite(cast(label_map_no_noise','uint8'),strcat(output_folder2,num2str(img_num),'.png'));
