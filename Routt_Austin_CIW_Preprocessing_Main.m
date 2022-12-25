% Routt, Austin
% Preprocess the Cells-In-Wells Dataset
% Saturday, March 16, 2022
clear all
close all
clc
%Set Random Seed for reproducibility
rng(13);
%% Adjust Parameters and Initialize the BlobAnalyser
% Images may require slight adjustments to these parameters

openRadius = 3; %Remove specks in binary mask with this pixel radius size - [px]
dilateRadius = 2; %Fill holes and make cells easier to seperate via watershed - [px]
spotSize = 1; %imextendedmin spot size in pixels, used on the distanceTransform image to get a mask for watershed - [px]
segImageSize = 170; %The required segnet input image size - [px^2]
maximumRBCCount = 99999999; %The maximum number of RBCs in an image - [RBCs]
minimumRBCArea = 100; %The minimum area, in pixels, of an RBC in the image - [px^2]
maximumRBCArea = 99999999999; %The maximum area, in pixels, of an RBC in the image - [px^2]

blobAreaRange = [minimumRBCArea maximumRBCArea]; %Min and Max rbc area as a range - [px^2]


%Initialize system object blobAnalyser
%Set MinimumBlobArea, MaximumBlobArea, and MaximumCount
System = struct(...
    'blobAnalyser', vision.BlobAnalysis('BoundingBoxOutputPort', true,'AreaOutputPort', true, 'CentroidOutputPort', true,'MinimumBlobArea',minimumRBCArea, 'MaximumBlobArea',maximumRBCArea,'MaximumCount', maximumRBCCount) ...
    );

%% Load Neural Nets and Label Categories

%Load the well semantic segmentation network
load('Routt_Austin_Deeplabv3_focal_tversky_WellsNotCells.mat', 'net');
Deeplabv3Wells = net;
clear net;

%Load the cell semantic segmentation network
load('Routt_Austin_Deeplabv3_CroppedWells_focal_tversky3.mat', 'net');
Deeplabv3Cells = net;
clear net;

%Load the mobilenetv2 with focal loss and scaling
load('Mobilenetv2\Routt_Austin_MobilenetV2_FocalLoss_Retrain_Ok.mat');
mobileNetV2 = net3;
clear net3

%Load the darknet with oversampling
load('Darknet\Routt_Austin_Darknet_Oversample_Retrain.mat');
darkNet = net2;
clear net2

%Load the shuffleNet with focal loss and scaling
load('Shufflenet\Routt_Austin_Shufflenet_FocalLoss_Retrain.mat');
shuffleNet = net3;
clear net3

%Load the Nasnetmobile
load('Nasnetmobile2\Routt_Austin_Nasnetmobile_FocalLoss_Retrain_best.mat');
nasnetMobile = net3;
clear net3


%Define cell morphology labels
label = categorical({'D','E1','E2','E3','S','SE','ST'});
%% Process all Cells-In-Wells Images

%Define directory addresses
baseImportDir = "";
baseExportDir ='Preprocessed Data';
%Define lists of donor, wash, and run folders
donors = {'Donor 4'};
washes = {'HSA'};
folder = '1';

%Note: If you have the complete dataset, use the following donors & washes
%Complete CIW dataset should be found here: https://doi.org/10.18738/T8/PSQKWF
%donors = {'Donor 1 11617', 'Donor 2 11718', 'Donor 3 11917', 'Donor 4 12417', 'Donor 5 12617', 'Donor 6 12917'};
%washes = {'HSA', Saline', 'SpinDown'};


%Iterate through all image directories (donor-wash-run folders)
for donor = donors
    for wash = washes
        %Create import directory string
        dirImport = fullfile(baseImportDir,donor{1},wash{1},folder);
        %Create export directory string
        dirExport = fullfile(baseExportDir,donor{1},wash{1},folder);
        %Create video export directory string
        dirVideo = fullfile(dirExport,'video.mp4');
        %Create image export directory string
        dirImages = fullfile(dirExport,'Images');
        %Create Mask export directory string
        dirMasks = fullfile(dirExport,'Masks');
        %Create Stats export directory string
        dirStats = fullfile(dirExport,'Stats');

        %Print the current import directory
        disp(dirImport);

        %Create the export directories if they don't exist
        if ~exist(dirExport, 'dir')
            mkdir(dirExport)
            mkdir(dirImages)
            mkdir(dirMasks)
            mkdir(dirStats)
        end


        %Create an image datastore for the image set
        imds = imageDatastore(dirImport, "IncludeSubfolders",false);



        % Go through images and determine the number of wells
        % Wells do not move, so the maximum number of well
        % bounding boxes can be used for all images in the set.

        %Set maxBboxesNum to 0
        maxBboxesNum = 0;

        %Set a different well search range for sequences that skip
        if (dirImport == fullfile(baseImportDir,donors{1},washes{1},folder))
            %Set the range to images after the jump in camera position
            range = 11:1:50;
        else
            %Set the image range to those that have different lighting
            range = 1:1:10;
        end

        % Start iterating through the set of images in range
        for i = range

            %Load the original image frame
            frame = imread(imds.Files{i});

            %Segment wells and get the bounding boxes
            [~, bboxes, ~] = Well_Seg4(frame, Deeplabv3Wells);

            %Check if number of bboxes is greater than maxBboxesNum
            %If greater than maxBboxesNum, update the maximum bboxes
            tempBboxesNum = height(bboxes);
            if tempBboxesNum > maxBboxesNum
                maxBboxesNum = tempBboxesNum;
                maxBboxes = bboxes;
            end
        end

        % Using the well bounding boxes, extract each well from every frame
        % and segment it using semantic segmentation.
        for i = 1:1:numel(imds.Files)
            %Track preprocessing progress
            progress = i/numel(imds.Files)*100;
            disp("Progress ~ "+num2str(progress)+"%");
            %Load the original image
            frame = imread(imds.Files{i});
            %Create a rgb color version of the frame for video testing
            frame2 = gray2rgb(frame);

            %Create a large blank mask to add cells from extracted well to
            lgMask = zeros(size(frame), 'logical');

            %Iterate through the bboxes and segment the cells in each well
            for j = 1:1:maxBboxesNum
                %Convert bbox coordinates to points
                points = bbox2points(maxBboxes(j,:));
                %Create row and column spans from points
                cols = points(1,1):1:points(2,1)-1;
                rows = points(1,2):1:points(3,2)-1;
                %Using spans, extract the well of interest
                wOI=frame(rows,cols);
                %Get the width and height of the well of interest
                [w, h] = size(wOI);
                %Make the extracted well of interest around 170x170 pixels
                n = max(0, ceil((segImageSize-w)/2));
                m = max(0, ceil((segImageSize-h)/2));
                wOI_Padded = padarray(wOI,[n m]); % pad

                %Resize to 224x224
                wOI_Padded_Lg = imresize(wOI_Padded, [224 224]);

                %Using semantic segmentation, segment the well of interest
                [C, ~] = semanticseg(wOI_Padded_Lg, Deeplabv3Cells);
                %Convert the categorical result into a black & white logical image
                BW = C == 'cell';
                %Resize to original padded image
                BW = imresize(BW, size(wOI_Padded));


                %Remove the padding around the binary image
                BW2 = BW(n+1:end-n,m+1:end-m); % unpad
                %Open the mask to remove small specks
                SE1 = strel('disk',openRadius);
                BW3 = imopen(BW2,SE1);
                %Dilate the image to fill small holes in cells & make them easier to seperate with watershed
                SE2 = strel('disk',dilateRadius,8);
                BW4 = imdilate(BW3,SE2);

                %Watershed step 1 - Get the negative distance transform of the inverse binary image
                distanceTransform = -bwdist(~BW4);

                %Watershed step 2 - Use imextendedmin to get small spots at the center of each blob
                mask0 = imextendedmin(distanceTransform,spotSize);

                %Watershed step 3 -Modify the distance transform, with the mask, to have local minima at the center of each blob
                distanceTransform2 = imimposemin(distanceTransform,mask0);

                %Watershed step 4 -Use the watershed transform to determine dividing lines based on the distance transform with imposed local minima
                waterShed = watershed(distanceTransform2);

                %Watershed step 5 -Apply dividing lines to the cleaned binary image
                BW5 = BW4;
                BW5(waterShed == 0) = 0;

                %Remove blob areas that are not within a given range
                BW6 = bwareafilt(BW5,blobAreaRange);

                %Get the areas, centroids, and bboxes for the cells in the well of interest
                [areas, centroids, bboxes] = System.blobAnalyser(BW6);
                %For the video test, annotate the cells in the well of interest
                if ~isempty(bboxes)
                    %Classify cell morphologies
                    cropSize = 227;

                    %Create a 51x51 blank image and set the background color to gray
                    background1 = ones(51, 'uint8')*0;


                    %Initialize storage variables
                    croppedImages1 = zeros(cropSize);  %Stores the cropped RBC images

                    %iterate through cell tracks and segment the individual cells
                    for ii= 1:1:height(bboxes)
                        %Set the current bounding box
                        currentBbox = bboxes(ii,:);
                        points = bbox2points(currentBbox);
                        colStart = max(1, points(1,1));
                        colEnd = max(2, points(2,1))-1;
                        rowStart = max(1, points(1,2));
                        rowEnd = max(2, points(3,2))-1;
                        col = colStart:1:colEnd;
                        row = rowStart:1:rowEnd;

                        %Crop the cell in the current bounding box, call it rbc
                        rbc = wOI(row, col);
                        %Reize
                        rbc = imresize(rbc,1.8);
                        [w, h] = size(rbc);
                        %Find the best position for rbc to be pasted onto a 51x51 blank image
                        startrow =  max(1, uint8((51 - w)/2));
                        startcol =  max(1, uint8((51 - h)/2));
                        a = background1;
                        b = rbc;
                        %Paste rbc into 51x51 image
                        a(startrow:startrow+size(b,1)-1,startcol:startcol+size(b,2)-1) = b;
                        %Resize to 227x227
                        a = imresize(a, [cropSize, cropSize]);
                        a = im2gray(a);      %make sure its a grayscale image
                        a = mat2gray(a); %Normalize


                        %Store all crops in the appropriate storage variable
                        croppedImages1 =cat(3,croppedImages1, a);   %Array of cropped RBCs (original)
                    end
                    %Remove the first initialization image, it's all black
                    croppedImages1(:,:,1) = [];
                    %Create storage variables for classifier softmax scores
                    %and morphology labels.
                    scores = {};
                    rbcType = categorical([]);
                    %Classify the cropped images using the maximum mean score 
                    for qq = 1:1:size(croppedImages1,3)
                        [~, score1] = classify(mobileNetV2,croppedImages1(:,:,qq));
                        [~, score2] = classify(darkNet,croppedImages1(:,:,qq));
                        [~, score3] = classify(shuffleNet,croppedImages1(:,:,qq));
                        [~, score4] = classify(nasnetMobile,croppedImages1(:,:,qq));
                        scores0 = [score1;score2;score3;score4];
                        scores(qq,:) = {scores0};
                        meanSoftmax = mean(scores0);
                        [~, I] = max(meanSoftmax);
                        rbcType(qq,:) =  label(I);
                    end
                    %Annotate the original image with a
                    %rectangle around each cell with morphology labels
                    originalImageAnnotated = insertObjectAnnotation(wOI,'rectangle',bboxes,rbcType,'TextBoxOpacity',0.9,'FontSize',10);

                else
                    %If there are no cells, then just convert to rgb
                    originalImageAnnotated = gray2rgb(wOI);
                end
                %Overlay the cell masks with annotated image
                overlayImg = labeloverlay(originalImageAnnotated,BW6);
                %Paste the annotated well into the rgb duplicate of frame
                frame2(rows,cols,:) = overlayImg;
                %Paste the well binary image into the large mask
                lgMask(rows, cols) = BW6;
                areasOfImage(j) = {areas};
                boxesOfImage(j) = {bboxes};
                scoresOfImage(j) = {scores};
                labelsOfImage(j) = {rbcType};
                centroidsOfImage(j) = {centroids};
            end

            %Add the original image to an array
            originalImages(:,:,i) = frame;
            %Add the annotated image to an array
            altImage(:,:,:,i) = frame2;
            %Add the large mask to an array
            masks(:,:,i) = lgMask;
            %Add the areas, bboxes, and centroids to an array
            areasOfImages(i) = {areasOfImage};
            centroidsOfImages(i) = {centroidsOfImage};
            boxesOfImages(i) = {boxesOfImage};
            scoresOfImages(i) = {scoresOfImage};
            labelsOfImages(i) = {labelsOfImage};

        end

        % Export the processed data
        video = VideoWriter(dirVideo,'MPEG-4'); %create the video object
        open(video); %open the file for writing
        for ii=1:numel(altImage(1,1,1,:)) %where N is the number of images
            writeVideo(video, im2frame(altImage(:,:,:,ii))); %write the image to file
        end
        close(video); %close the file

        %Save images & masks
        for i = 1:1:length(originalImages(1,1,:))
            n_strPadded = sprintf( '%06d', i ) ;

            %Create the base filename
            baseFileName1 = strcat("image",'_',num2str(n_strPadded),'.png');
            baseFileName2 = strcat("mask",'_',num2str(n_strPadded),'.png');

            %Set image, mask, and label full names
            imageFullFileName = fullfile(dirImages, baseFileName1);
            maskFullFileName = fullfile(dirMasks, baseFileName2);

            %Save image
            imwrite(originalImages(:,:,i), imageFullFileName);
            %Save mask
            imwrite(masks(:,:,i), maskFullFileName);
        end
        % Save bboxes, areas, and centroids
        stats=table(boxesOfImages', scoresOfImages', labelsOfImages',areasOfImages',centroidsOfImages', 'VariableNames',{'Bboxes', 'Scores','Labels','Areas','Centroid'});

        baseFileName3 = strcat("stats",".mat");
        baseFileName4 = strcat("Well_Bboxes",".mat");
        statsFullFileName = fullfile(dirStats, baseFileName3);
        wellsFullFileName = fullfile(dirStats, baseFileName4);
        save(statsFullFileName,"stats");
        save(wellsFullFileName,"maxBboxes");

    end
end

disp("Processing Complete");