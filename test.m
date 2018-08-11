tic
for n = 1299:1514
    slic_stand_alone_fn(n)
end
toc

%%
output_folder2 = '/home/pkr/pkr-work/pose_estimation/build/segmentlabels/';

images = dir(output_folder2);

  