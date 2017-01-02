function predict_label = your_kNN(feat)
% Output should be a fixed length vector [num of img, 1]. 
% Please do NOT change the interface.
sz = size(feat);
model = load('model');
mod_idf = log(model.idf);
mod_tf = model.tf;
feat_desc = double(model.feat_train);
fdesSz = size(feat_desc);



for i = 1 : sz(1)
    %euclidiand distance
    euclDist = pdist2(feat(i , :),feat_desc);
    [m , id] = min(euclDist);
    
    %{
    %using tf_idf
    
    %tf of query image
    tfQ = curFeature ./ sum(curFeature);
   
     for j = 1 : fdesSz(1)
         tfD = mod_tf(j , :);
            diff = (tfQ - tfD).* mod_idf;
         temp(1, j ) = sqrt(diff * diff');
     end;
     [m , id] = min(temp);
     %}
    
    %cosin between features
    %{
     for j = 1 : fdesSz(1)
         
         % tfD =  mod_tf(j , :) .* mod_idf;%(curFeature >= 0) .*
        
         %temp(1, j ) = dot(tfQ , tfD) / (norm(tfQ) * norm(tfD));
         temp(1, j ) = dot(feat(i , :) , feat_desc(j , :)) / (norm(feat(i , :)) * norm(feat_desc(j , :)));
          %temp(1, j ) = dot(curFeatureTemp , feat_descTemp) / (norm(curFeatureTemp) * norm(feat_descTemp));
          % temp(1, j ) = dot(curFeature , tf_idf(j , :)) / (norm(curFeature) * norm(tf_idf(j , :)));
     end;
             
  [m , id] = max(temp);
    
   %}
     label = model.label_train(id);
    predict_label(i , 1) = label;%zeros(size(feat,1),1); %dummy. replace it with your own code
end;
end