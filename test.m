P = 100;
size_label_map2 = size(label_map2);
label_map_no_noise = zeros(size_label_map2(1),size_label_map2(2));
new_cluster_index = 1;
for i=1:K
    [Xa,Ya] = find(label_map2 == i);
    BW = false(size_label_map2(1),size_label_map2(2));
    for p=1:length(Xa)
        BW(Xa(p),Ya(p)) = true;
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
cIndMap = uint16(cIndMap);

imwrite(imgVis,strcat(num2str(img_num),'_segment_clustered.png'));

imwrite(cast(label_map_no_noise','uint8'),strcat(num2str(img_num),'_segment_map.png'));
