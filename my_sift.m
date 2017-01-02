function [feat ,po] = my_sift(img)

    
   [iR , iC , l] = size(img);
        po = 0;
    
    sigma = 2.5;
    gusFilter = fspecial('gaussian', 16 , sigma);
  
    %find interset points
    surfFeatures = detectSURFFeatures(img);
    [numOfFeat , l] = size(surfFeatures);
 
    feat2Dim = 128;
  
    
    feat = zeros(numOfFeat , feat2Dim);
    %extract local feature descriptor for each features detected
    for fea = 1: numOfFeat
        fPos = surfFeatures(fea).Location;
     
        %get x y coordintes of  patch around interset point
        fpRowS = uint16(fPos(1) - 8) ;
        fpRowE = uint16(fPos(1) + 7) ;
        fpColS = uint16(fPos(2) - 8) ;
        fpColE = uint16(fPos(2) + 7) ;
        
        %handle boundry conditions
        if(fpRowS < 1 )
            fpRowS = 1;
        end;
        if(fpRowE > iR)
            fpRowE = iR;
        end;
        
        if(fpColS < 1 )
            fpColS = 1;
        end;
        if(fpColE > iC)
            fpColE = iC;
        end;
        
        %img patch around feature
        patch = img(fpRowS : fpRowE , fpColS :fpColE);  
        psz = size(patch);
        if(psz(1) < 16)
            patch(psz(1) + 1 : 16 , :) = 0;
        end;
        if(psz(2) < 16)
            patch(: , psz(2) + 1: 16 ) = 0;
        end;
        
        %image gradient
        [gMag,gDir] = imgradient(patch);
        gDir = gDir + 180;
        
        [dirHist,angles] =  imhist(patch, 36);
        [dHist,idR]= sort(dirHist , 'descend');
       tr = angles(idR(1)) * .8 ;
       
       %find dominant oriantation
       count = 1;
       domAngle = 0;
       while(angles(idR(count)) >= tr && count  < 11)
           %disp(count);
           domAngle  = domAngle + angles(idR(count));
           count =  count + 1;
       end;
       domAngle = domAngle / count;
        
        
        patch = gusFilter .* gDir;
        
        %find histogram for feature descriptors 
        for rw = 1 : 4                      %4 windows around feature in row
            for cw = 1 : 4                  %4 windows around feature in col
                %featRange = ((rw - 1) *  8 ) * 4 + (cw - 1) *  8;
                for rp = 1 : 4              %4 pixel in each window for row
                    for cp = 1 : 4          %4 pixel in each window for col
                       r = (rw - 1) * 4 + rp;     %row offset for patch
                       c = (cp - 1) * 4 + cp;     %col offset for patch
                       
                       %f = 1;               %offset for 128D feature vector
                       dir = patch(r, c);
                       %dir = (dir / pi)  * 180; %already convereted
                       dir = mod((dir - domAngle) , 360);
                       fhBin = -1;
                       if ((dir >= 337.5 || dir < 22.5)) % could remove two extra condition . (is grad dir is between 0 and 360?)
                           fhBin = 1;   %0
                       elseif (dir >= 22.5 && dir < 67.5)
                           fhBin = 2;   %45
                       elseif (dir >= 67.5 && dir < 112.5)
                           fhBin = 3;   %90
                       elseif (dir >= 112.5 && dir < 157.5)
                           fhBin = 4;   %135
                       elseif (dir >= 157.5 && dir < 202.5)
                           fhBin = 5;   %180
                       elseif (dir >= 202.5 && dir < 247.5)
                           fhBin = 6;   %225
                       elseif (dir >= 247.5 && dir < 292.5)
                           fhBin = 7; %270
                       elseif (dir >= 292.5 && dir < 337.5)
                           fhBin = 8; %315
                       end;
                        %featOffset = featRange + fhBin;
                        feat(fea , fhBin) =   feat(fea , fhBin) + 1;
                    end;
                end;
            end;
        end;
    end;

    
end