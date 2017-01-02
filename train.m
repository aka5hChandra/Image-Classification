% You can change anything you want in this script.
% It is provided just for your convenience.
clear; clc; close all;

img_path = './train/';
class_num = 30;%30;
img_per_class = 60;
img_num = class_num .* img_per_class;
%feat_dim = size(feature_extraction(imread('./val/Balloon/329060.JPG')),2);
feat_dim = 64 ;%+ 255;%128;%
folder_dir = dir(img_path);
k = 100;%62;%62;%43;%uint8(sqrt(feat_dim));%43
feat_train = zeros(img_num,k);
tf = zeros(img_num,k);
idf = zeros(1,k);
label_train = zeros(img_num,1); 


feature_descriptor = [];
descOffsetList = [1];
descOffset = 1;

for i = 1:length(folder_dir)-2
    
    img_dir = dir([img_path,folder_dir(i+2).name,'/*.JPG']);
    if isempty(img_dir)
        img_dir = dir([img_path,folder_dir(i+2).name,'/*.BMP']);
    end
    
    label_train((i-1)*img_per_class+1:i*img_per_class) = i;
    
    for j = 1:length(img_dir) 
        img = imread([img_path,folder_dir(i+2).name,'/',img_dir(j).name]);
         
            [iR , iC , l] = size(img);
            ori = img;
            if(l == 3)
                img = rgb2gray(img);
            end;
            %Detect interest points
            surfPoints = detectSURFFeatures(img);
            
            %extract feature 
            [features, valid_points] = extractFeatures(img, surfPoints  , 'Method' , 'SURF' , 'SURFSize' , 64);%my_sift(img);%
            fsize = size(features);
            
            %color as feature % not using it
            %{%
            bin = 24;
            [rHist,binLocations] =  imhist(ori(:,:,1) , bin);
            [gHist,binLocations] =  imhist(ori(:,:,2), bin);
            [bHist,binLocations] =  imhist(ori(:,:,3), bin);
            %[rHist,idR]= sort(rHist);
            %[gHist , idG]= sort(gHist);
            %[bHist , idB] = sort(bHist);
            %coloHist = [sum(rHist( idR(250 : end , 1), 1 )) , sum(gHist(idG(250 : end , 1), 1)) , sum(bHist(idB(250 : end , 1)))];
            %ori = rgb2hsv(ori);
            %[hHist,binLocations] =  imhist(ori(:,:,1));
            coloHist = [rHist ; gHist ; bHist];
            coloHist = log(coloHist);
            coloHist((coloHist == -Inf)) = 0;
            coloHistMat = repmat(coloHist' , fsize(1) , 1);%tranpse
            colorExtedendFeature = [features(1 : fsize(1) , :) , coloHistMat ];
            %}
            
            %store features of all images
            feature_descriptor(descOffset : descOffset + fsize(1) - 1 , :) = colorExtedendFeature;%features(1 : fsize(1) , :);
            
            %offset in array of features for futhrer refreance
            descOffset = descOffset + fsize(1);
            descOffsetList = [descOffsetList , descOffset]; %or make all features of same size
    end
end

%Generate Codebook
[idK , codeBook] = kmeans(feature_descriptor , k  , 'MaxIter', 1000);

save('codeBook.mat' , 'idK' ,  'codeBook');
descOffsetLen = size(descOffsetList , 2);

%Genereate feature descriptor
for im = 1 : img_num(1) 
    descStart = descOffsetList(1 , im);
    descEnd = descOffsetList(1 , im + 1) - 1;
    for d = descStart : descEnd
        id = idK(d, 1);
        feat_train(im , id) = feat_train(im , id) + 1;
    end;
  
end;

%tf
for im = 1 : img_num(1)
    nd = sum(feat_train(im , :));
     tf(im , :) = feat_train(im , :)./nd;
     %{
     tf(im , :) = log( feat_train(im , :));%./nd;
     tf((tf == -Inf)) = 0;
     tf = 1 +tf;
     %}
   
end;

%idf
 %for id = 1 : k     
        ni = sum(feat_train(: , id) > 0);
        idf_ = log(img_num / ni);
        idf_((idf_ == -Inf)) = 0;
        idf =  idf_;%tf_ * idf;
%end;

%save('model.mat','feat_train','label_train');
save('model.mat','feat_train','label_train' ,'idf' , 'tf');