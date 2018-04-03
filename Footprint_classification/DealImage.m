%% for image in problem 1
for i = 1:10000
    str1 = strcat('./data_test/1_test/',int2str(i),'.png');
    k = imread(str1);
    k = RotationCorrection(k);
    k = imgaussfilt(k,0.8);
    k = wiener2(k,[5 5]);
    k = imbinarize(k,0.65);
    k = medfilt2(k,[2,2]);
    k = medfilt2(k,[2,2]);

    k = imresize(k,[300,100]);

    str2 = strcat('./data_test/1_test_deal/',int2str(i),'.png');
    imwrite(k,str2);
end
%%

%% for problem 2, train image, do average binarize and filter
for i = 1000:1000
    str1 = strcat('./2/train/0',int2str(i),'.png');
    k = imread(str1);
    k = imgaussfilt(k, 0.7);

    k = imbinarize(k,'adaptive','ForegroundPolarity','dark');
    k = uint8(k)*255;
    k = imgaussfilt(k, 1);
    k = imresize(k,[125,64]);
    str2 = strcat('./2_kmeans/train/',int2str(i),'.png');
    imwrite(k,str2);
end
%%


%% for valid image %%
for i = 99
    str1 = strcat('./3/000',int2str(i),'.jpg');
    k = imread(str1);
    k = RotationCorrection(k);
    k = medfilt2(k,[2,2]);
    k = imgaussfilt(k, 1);
    k = imbinarize(k,0.4);
    k = medfilt2(k,[2,2]);
%     k = wiener2(k,[2 2]);
%     k = imresize(k,[300,100]);
    str2 = strcat('./3_test/',int2str(i),'.jpg');
    imwrite(k,str2);
end

