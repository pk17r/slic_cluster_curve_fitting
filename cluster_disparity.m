
input_folder = 'input/';
img_num = 1239;
I = imread(strcat(input_folder,num2str(img_num),'d.png'));
img = cast(img, 'uint16');
size_C2 = size(C2);

for i=1:size_C2(1)
    [Xc,Yc] = find(label_map_no_noise' == i);
    Zp = zeros(length(Xc),1);
    Xp = Zp;
    Yp = Zp;
    counts = 1;
    for p=1:length(Xc)
        if Xc(p) > 20 && Xc(p) < 700 && Yc(p) > 160 && Yc(p) < 1260
            Zp(counts) = img(Xc(p),Yc(p));
            Xp(counts) = Xc(p);
            Yp(counts) = Yc(p);
            counts = counts + 1;
        end
    end
    counts = counts - 1;
    Zc_counts = Zp(1:counts);
    Xc_counts = Xp(1:counts);
    Yc_counts = Yp(1:counts);
    %z = ceil(mean(Zc_counts));
    
    %B = [ones(counts,1), yourData(:,1:2)] \ yourData(:,3);
    A = [Xc_counts, Yc_counts, ones(counts,1)];
    B = A \ Zc_counts;
    
    for p=1:length(Xc)
        img(Xc(p),Yc(p)) = (B(1) * Xc(p) + B(2) * Yc(p) + B(3)) * 200;
    end
end


imwrite(img,strcat(output_folder,num2str(img_num),'d_planefitted.png'));