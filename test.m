cluster_D_threshold = 14.6;
C2 = zeros(K,6);
C2(:,1:3) = C(:,1:3);

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
    C2(i,4) = lstd;
    C2(i,5) = astd;
    C2(i,6) = bstd;
end
