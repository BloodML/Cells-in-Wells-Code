function [mask, bboxes, centroids] = Well_Seg4(frame,net)
%This function takes the first frame and uses threshold segmentation with
% the regionprops function to output the mask, bounding box, and centroids for all wells

% Inputs:
% frame = the first frame in the video - [-]


% Outputs:
%mask = binary images for frame - [-]
%bboxes = bounding boxes for all wells - [-]
%centroids = centroids for all wells - [-]

%Set constants and setup blob analyzer
maximumCount = 99999999; %The maximum number of RBCs in an image - [RBCs]
minimumArea = 9000; %The minimum area, in pixels, of an RBC in the image - [px^2]
maximumArea = 30000; %The maximum area, in pixels, of an RBC in the image - [px^2]
radius = 25; %Remove specks in binary mask with this pixel radius size - [px]

System = struct(...
    'blobAnalyser', vision.BlobAnalysis('BoundingBoxOutputPort', true,'AreaOutputPort', false, 'CentroidOutputPort', true,'MinimumBlobArea',minimumArea, 'MaximumBlobArea',maximumArea,'MaximumCount', maximumCount) ...
    );

%Resize to 1/2
frame0 = imresize(frame,1/2);

%Convert the frame from grayscale to rgb
frame1 = gray2rgb(frame0);

mask0 = semanticseg(frame1,net);
mask1=mask0 == 'cell';
mask2 = imresize(mask1,2);

se = strel('disk',radius);
mask = imopen(mask2,se);

%Perform blob analysis on the segmented binary image to find area, centroids, and bounding boxes for each cell.
[centroids, bboxes] = System.blobAnalyser(mask);


end

