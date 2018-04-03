function correctedImage = RotationCorrection(im)
% This function gets a rotated image and corrects its clockwise or
% counterclockwise rotation. 

%% Edge detection and edge linking

binaryImage = imgaussfilt(im,1);
binaryImage = imbinarize(binaryImage,0.99);
binaryImage = uint8(binaryImage)*255;

binaryImage = edge(binaryImage,'canny'); % 'Canny' edge detector
binaryImage = bwmorph(binaryImage,'thicken'); % A morphological operation for edge linking
%%
 
%% Radon transform to compute rotation angles
theta = -90:89;
[R,xp] = radon(binaryImage,theta);
[R1,r_max] = max(R); 
 
[~, theta_max] = max(R1);
correctedImage = imrotate(im,-theta_max+90,'crop'); % Rotation correction
correctedImage(correctedImage == 0) = 255; % Converts black resgions to white regions
%%