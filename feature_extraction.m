function feat = feature_extraction(img)
% Output should be a fixed length vector [1*dimension] for a single image. 
% Please do NOT change the interface.
    
    k = uint8(100); %43
    %find a way to do load only once
    cb = load('codeBook');
    feat = zeros(1 , k);
    [iR , iC , l] = size(img);
    
    %color as feature
    %{%
      bin = 24;
    [rHist,binLocations] =  imhist(img(:,:,1), bin);
    [gHist,binLocations] =  imhist(img(:,:,2), bin);
    [bHist,binLocations] =  imhist(img(:,:,3), bin);
    coloHist = [rHist ; gHist ; bHist];
    coloHist = log(coloHist);
    coloHist((coloHist == -Inf)) = 0;
    %}
       
    if(l == 3)
        img = rgb2gray(img);
    end;
 
    surfPoints = detectSURFFeatures(img);
    
    [features, valid_points] = extractFeatures(img, surfPoints , 'Method' , 'SURF' , 'SURFSize' , 64);% my_sift(img);% 
  
    
    fsize = size(features);
    features = double(features);
    
        for i = 1 : size(features , 1)
            lfeat = features(i , :);
            lfeat = [features(i , :), coloHist'];
            %Euclidian distance
            euclDist = pdist2(lfeat,cb.codeBook);
            [m , id] = min(euclDist);%euclDist%cosTheta%temp
            feat(1 , id) =  feat(1 , id) + 1;
        end;
    
end