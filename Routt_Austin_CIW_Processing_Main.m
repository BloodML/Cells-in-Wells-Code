% Routt, Austin
% Processing the Cells-In-Wells Dataset - Track Cells & Analyze the Data
% Saturday, March 20, 2022
clear all
close all
clc

%Set Random Seed for reproducibility
rng(13);
%% Tracking 

% Load Preprocessed images

%Define directory address primitives
baseDir = 'Preprocessed Data';
donors = {'Donor 4'};
washes = {'HSA'};
folder = '1';
%donors = {'Donor 1 11617', 'Donor 2 11718', 'Donor 3 11917', 'Donor 4 12417', 'Donor 5 12617', 'Donor 6 12917'};
%washes = {'HSA','Saline', 'SpinDown'};



%Iterate through all image directories
for donor = donors
    for wash = washes
        %Create import directory string
        dirImport = fullfile(baseDir,donor{1},wash{1},folder);
        %Create export directory string
        dirExport = fullfile("Processed Data",donor{1},wash{1},folder);
        %Create image import directory string
        dirImages = fullfile(dirImport,'Images');
        %Create Mask import directory string
        dirMasks = fullfile(dirImport,'Masks');
        %Create Stats import/export directory string
        dirStats = fullfile(dirImport,'Stats',"stats.mat");
        dirWellBboxes = fullfile(dirImport, "Stats/Well_Bboxes.mat");
        dirStatsEx = fullfile(dirExport,'Stats');

        %Create the export directories if they don't exist
        if ~exist(dirExport, 'dir')
            mkdir(dirExport)
            mkdir(dirStatsEx)
        end

        %Print the current import directory
        disp(dirImport);



        %Create an image datastore for the image set
        imds = imageDatastore(dirImages);

        % Create a pixelLabelDatastore for the ground truth pixel labels.
        classNames = ["background", "cell"];
        labelIDs   = [0 1];
        pxds = pixelLabelDatastore(dirMasks,classNames,labelIDs);

        %Load the stats and well bboxes
        load(dirStats, 'stats');
        load(dirWellBboxes, 'maxBboxes');

        %Export stats and well bboxes
        save(strcat(dirStatsEx, "/Well_Bboxes.mat"),'maxBboxes');
        save(strcat(dirStatsEx, "/stats.mat"),'stats');

        %% Visualize training images and ground truth pixel labels.
        I =  imread(imds.Files{10});
        BW1 = imread(pxds.Files{10});
        originalMI = labeloverlay(I,BW1);
        figure
        imshow(originalMI)

        %% Initialize system objects, data structures, and constants
        System = struct(...
            'videoPlayer', vision.DeployableVideoPlayer('Size', 'Custom', 'CustomSize', [1280 720]));

        %Get the bounding boxes of all wells in the first frame
        All_Wells_Bboxes = maxBboxes;
        %Generate names for all wells
        wellNames = {};
        for ee = 1:1:height(All_Wells_Bboxes)
            wellNames{ee} = strcat("Well ", num2str(ee));
        end
        %Keep track of the well number
        wellNum = 0;
        for well = wellNames
            disp(well{1});
            %increment the well number
            wellNum = wellNum +1;

            % Create an empty structure, with appropriate fields, to track cells and classify them.
            Cells = struct(...
                'id', {}, ...                       %To maintain a consistent identity between cells
                'bbox', {}, ...                     %To show tracking in the frame
                'frameN',{}, ...                    %To associate centroids with the frame number they were found in
                'centroidHistory',{},...            %To analyze incremental displacements
                'kalmanFilter', {}, ...             %To assign specific movements to specific cells and make precitions on future movements
                'softmaxHistory',{}, ...            %To know the evolving probability distribution
                'labelHistory',{}, ...              %To know the morphology categorical time series
                'avgSoftmaxHistory',{}, ...         %To get the ensemble softmax scores
                'avgLabelHistory', {}, ...          %To get the label associated with the highest softmax score
                'age', {}, ...                      %To know how old the track is; used primarily to help determine frame removal
                'totalVisibleCount', {}, ...        %To know the total number of frames the cell was visible; helps determine if the cell track should be deleted
                'consecutiveInvisibleCount', {});   %To know the number of consecutive frames the cell was invisible; helps determine if the cell track should be deleted


            %Create an "ID of the next cell" variable for future unassigned cells
            nextId = 1;


            %Declare the name order variable
            name_order = {};   %To associate unique names with each cell's ID number

            %Create the well export directory string
            dirWell = fullfile(dirExport,well{1});
            %Create video export directory string
            dirVideo = fullfile(dirWell,'video.mp4');


            %Create the export directories if they don't exist
            if ~exist(dirWell, 'dir')
                mkdir(dirWell)
            end

            % Export video
            video = VideoWriter(dirVideo,'MPEG-4'); %create the video object
            open(video); %open the file for writing


            %Set Random Seed for reproducibility
            rng(wellNum);
            %set the well of interest
            Well_of_Interest = All_Wells_Bboxes(wellNum,:);

            %Get the rows and columns that span the well of interest
            points = bbox2points(Well_of_Interest);
            %Create row and column spans from points
            cols = points(1,1):1:points(2,1)-1;
            rows = points(1,2):1:points(3,2)-1;
          

            %Detect cells, and track them across frames.
            %Loop through the video frames

            for frameCount = 1:1: height(imds.Files)


                % Get the current frame and mask
                Current_Frame = imread(imds.Files{frameCount});
                BW0 = imread(pxds.Files{frameCount});

                %Highlight the well of interest, with a red box, on the current frame
                Current_Frame_Annotated = insertShape(Current_Frame,'rectangle',Well_of_Interest,'LineWidth',10,'Color','red');
                Current_Frame_Annotated = labeloverlay(Current_Frame_Annotated, BW0);

                %Crop out the well of interest and resize to a 175x175x3 image
                %Note, this will require a different px to um conversion factor for
                %each well - This is due to the image training set
                Well_Frame0 = Current_Frame(rows,cols);

                %Segment the well frame into a binary image, where cells are white
                %blobs on a black background; uses semantic segmentation and watershed

                BW1 = BW0(rows,cols);

                %Resize the well frame and corresponding binary image
                %image
                Well_Frame = imresize(Well_Frame0, 1);
                BW = imresize(BW1, 1);

                %Perform blob analysis on the segmented binary image to find centroids
                %and bounding boxes for each cell.
                centroids = stats.Centroid{frameCount,1}{1,wellNum};
                bboxes = stats.Bboxes{frameCount,1}{1,wellNum};
                scores = stats.Scores{frameCount,1}{1,wellNum};
                labels = stats.Labels{frameCount,1}{1,wellNum};
                %[~, centroids, bboxes] = System.blobAnalyser(BW);

                %Track the cells
                [Cells,nextId] = Routt_Austin_Tracker3(Cells,nextId,frameCount, centroids, bboxes, scores, labels);

                %Set the classification image size
                cropSize = 227;

                %Create a 51x51 blank image and set the background color to gray
                background1 = ones(51, 'uint8')*0;
                background2 = ones(51, 'uint8')*220;

                croppedImages1 = [];
                %Initialize storage variables
                %iterate through cell tracks and segment the individual cells
                for i= 1:1:length(Cells)
                    %Set the current bounding box
                    currentBbox = Cells(i).bbox;
                    %Set the current label
                    if(~isempty(Cells(i).avgLabelHistory))
                        currentLabel = Cells(i).avgLabelHistory(end);
                    else
                        currentLabel = cellstr('-');
                    end
                    %Set mean softmax score
                    if(~isempty(Cells(i).avgSoftmaxHistory))
                        meanSoftmax = Cells(i).avgSoftmaxHistory(end, :);
                    else
                        meanSoftmax = 0;
                    end

                    points = bbox2points(currentBbox);
                    colStart = max(1, points(1,1));
                    colEnd = max(4, points(2,1))-1;
                    rowStart = max(1, points(1,2));
                    rowEnd = max(3, points(3,2))-1;

                    if rowEnd>length(rows)
                        rowEnd = length(rows);
                    end
                    if colEnd>length(cols)
                        colEnd=length(cols);
                    end

                    col = colStart:1:colEnd;
                    row = rowStart:1:rowEnd;

                    %Crop the cell in the current bounding box, call it rbc
                    rbc = Well_Frame(row, col);

                    if(isempty(rbc))
                        break;
                    end



                    %Reize
                    rbc = imresize(rbc,1.8);
                    %Find height and width of rbc image
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


                    box_color = 'green';
                    position = [10 20];
                    text_str = strcat(char(currentLabel),': ', num2str(max(meanSoftmax)*100));
                    img = insertText(a,position,text_str,'FontSize',19,'BoxColor', box_color,'BoxOpacity',0.3,'TextColor','white');

                    %Store all crops in the appropriate storage variable
                    croppedImages1(:, :, :, i) = img;   %Array of cropped RBCs (original)
                end




                h3 = imtile(croppedImages1)./255;

                infoPic0 = ones([480 640], 'double')*0;
                box_color = 'red';
                text_str = strcat("Donor: ", donor{1} );
                infoPic1 = insertText(infoPic0,[100 20],text_str,'FontSize',30,'BoxColor', box_color,'BoxOpacity',0.3,'TextColor','white');
                text_str = strcat("Wash: ", wash{1} );
                infoPic1 = insertText(infoPic1,[100 120],text_str,'FontSize',30,'BoxColor', box_color,'BoxOpacity',0.3,'TextColor','white');
                text_str = strcat("Well: ", well{1} );
                infoPic1 = insertText(infoPic1,[100 220],text_str,'FontSize',30,'BoxColor', box_color,'BoxOpacity',0.3,'TextColor','white');
                text_str = strcat("Frame Count: ", num2str(frameCount) );
                infoPic1 = insertText(infoPic1,[100 320],text_str,'FontSize',30,'BoxColor', box_color,'BoxOpacity',0.3,'TextColor','white');

                %Create a final frame that shows the current frame and the well of
                %interest, with cells being tracked
                [name_order,out] = Routt_Austin_Display3(Cells,name_order, ...
                    Current_Frame_Annotated, Well_Frame, h3, infoPic1);



                writeVideo(video, im2frame(out)); %write the image to file


                % Display the final frame in the video player
                System.videoPlayer.step(out);





            end
            %Release the System objects
            release(System.videoPlayer);
            close(video); %close the file

            %Export Cell tracks
            baseFileName = strcat("Cell_Tracks",".mat");
            tracksFullFileName = fullfile(dirWell, baseFileName);
            save(tracksFullFileName,"Cells");

        end
    end
end


disp('Problem 1 end')
disp('-----------------------------------------------------------')



%% Data Analysis - Collect the cell track data & export it

%Define directory address primitives
baseDir = 'Processed Data';
donors = {'Donor 4'};
washes = {'HSA'};
folder = '1';

%Use these if you have the full dataset
%baseDir = 'D:\Dataset_Retrained';
%donors = {'Donor 1 11617', 'Donor 2 11718', 'Donor 3 11917', 'Donor 4 12417', 'Donor 5 12617', 'Donor 6 12917'};
%washes = {'HSA','Saline', 'SpinDown'};
%folder = '1';

%Create the export directory
dirExport = fullfile(baseDir, 'Data Analysis');
if ~exist(dirExport, 'dir')
    mkdir(dirExport)
end

%Create a container for all cell tracks
allCellTracks2 = {};
%Initialize the wash & cell counts to 0
washNum = 0;
cellNum = 0;
%Iterate through the washes cell
for wash = washes
    %Increment the wash count
    washNum = washNum +1;
    %Initialize the donor count to 0
    donorNum = 0;
    %Iterate through the donors cell
    for donor = donors
        %Increase the donor count by 1
        donorNum = donorNum+1;
        %Create import directory string
        dirImport = fullfile(baseDir,donor,wash,folder);
        dirWellBboxes = fullfile(dirImport, "Stats/Well_Bboxes.mat");


        %Print the current import directory
        disp(dirImport);

        %Load the stats and well bboxes
        load(dirWellBboxes, 'maxBboxes');

        %Get the bounding boxes of all wells
        All_Wells_Bboxes = maxBboxes;

        %Generate names for all wells
        wellNames = {};
        for ee = 1:1:height(All_Wells_Bboxes)
            wellNames{ee} = strcat("Well ", num2str(ee));
        end

        %Keep track of the well number
        wellNum = 0;
        %Iterate through the wells
        for well = wellNames
            %Increment the well number
            wellNum = wellNum+1;
            %Keep track of cell number
            %Create cell track import directory string
            dirCells = fullfile(dirImport,well{1}, "Cell_Tracks.mat");
            %Load the cell tracks
            load(dirCells, 'Cells');
            %Initialize the well cell count
            wellCellNum = 0;
            for cell = Cells
                % for each cell make sure the number of frames in cell track is greater than 100

                %Increment cell number and wellCellNum
                cellNum = cellNum+1;
                wellCellNum = wellCellNum+1;
                allCellTracks2(cellNum, :) = {
                    char(wash{1}),...
                    char(donor{1}),...
                    char(well{1}),...
                    Cells(wellCellNum).id,...
                    Cells(wellCellNum).frameN,...
                    Cells(wellCellNum).centroidHistory,...
                    Cells(wellCellNum).softmaxHistory,...
                    Cells(wellCellNum).labelHistory,...
                    Cells(wellCellNum).avgSoftmaxHistory, ...
                    Cells(wellCellNum).avgLabelHistory, ...
                    Cells(wellCellNum).age, ...
                    Cells(wellCellNum).totalVisibleCount, ...
                    Cells(wellCellNum).consecutiveInvisibleCount, ...
                    Cells(wellCellNum).bbox
                    };

            end


        end


    end


end

baseFileName = strcat("allCellTracks2",".mat");
cellsFullFileName = fullfile(dirExport, baseFileName);

save(cellsFullFileName,"allCellTracks2");

%% Load the cell track data

clear all
close all
clc

%Define directory address primitives
baseDir = 'Processed Data';

%Create the export directory
dirExport = fullfile(baseDir, 'Data Analysis');

baseFileName = strcat("allCellTracks2",".mat");
cellsFullFileName = fullfile(dirExport, baseFileName);

load(cellsFullFileName,"allCellTracks2");

T = cell2table(allCellTracks2,...
    "VariableNames",["Wash", "Donor" ,"Well", "CellWellID","Frames", "CentroidHistory","SoftmaxHistory", "LabelHistory", "AvgSoftmaxHistory", "AvgLabelHistory", "Age", "TotalVisibleCount", "ConsecutiveInvisibleCount", "Bboxes"]);
dataTable = T;
dataFullFileName = fullfile(dirExport, "dataTable");
save(dataFullFileName,"dataTable");



%% Get the data for Analysis of Shape Change Matrices, MIs, and Anomalies


washLabels = categorical([]);
washDonorLabels = categorical([]);
washTables = [];
washDonorTables = [];
anoms = [];

donors = {'Donor 4'};
washes = {'HSA'};
folder = '1';

%Define wash labels and initialize the wash count
%washes = {'HSA','Saline', 'SpinDown'};
%donors = {'Donor 1 11617', 'Donor 2 11718', 'Donor 3 11917', 'Donor 4 12417', 'Donor 5 12617', 'Donor 6 12917'};
washCount = 0;


%Iterate through the wash labels
for wash = washes
    %Increase the wash count
    washCount = washCount+1;
    %Create a wash subset
    washTables = dataTable(strcmp(dataTable.Wash, wash{1}), :);
    %Initialize the donor count
    donorCount = 0;
    %Iterate through the donors to separate
    for donor = donors
        %Increase the donor count
        donorCount = donorCount+1;
        %Create a donor subset
        washDonorTables = washTables(strcmp(washTables.Donor, donor{1}), :);
        %Iterate through the cells
        for nn = 1:1:height(washDonorTables)
            %Only use cells with 100 frames or more because the window size is 100 frames
            if height(washDonorTables.Frames{nn})>=100
                %Get the cell's frames from element 100 and beyond
                frames = washDonorTables.Frames{nn}(100:end);
                %Associate the frames with the moving average labels
                %Because of the moving window, the first label is the 100th frame
                washDonorLabels(frames, nn, washCount, donorCount) = washDonorTables.AvgLabelHistory{nn};
            end
        end

    end
    %Iterate through the cells in the wash
    for ii = 1:1:height(washTables)
        %Only use cells with 100 frames or more because the window size is 100 frames
        if height(washTables.Frames{ii})>=100
            %Get the cell's frames from element 100 and beyond
            frames = washTables.Frames{ii}(100:end);
            %Associate the frames with the moving average labels
            %Because of the moving window, the first label is the 100th frame
            washLabels(frames, ii, washCount) = washTables.AvgLabelHistory{ii};
        end

    end
    %From the wash, extract the 100th and 600th frame's moving average label
    wellLabels_change = [washLabels(100, :, washCount)', washLabels(600, :, washCount)'];

    %Make a table of all anomolous transitions
    anom_indx = (wellLabels_change(:, 1) == 'SE') & (wellLabels_change(:, 2) == 'E3')|...
        (wellLabels_change(:, 1) == 'SE') & (wellLabels_change(:, 2) == 'E2')|...
        (wellLabels_change(:, 1) == 'SE') & (wellLabels_change(:, 2) == 'E1')|...
        (wellLabels_change(:, 1) == 'SE') & (wellLabels_change(:, 2) == 'D')|...
        (wellLabels_change(:, 1) == 'S') & (wellLabels_change(:, 2) == 'SE')|...
        (wellLabels_change(:, 1) == 'S') & (wellLabels_change(:, 2) == 'E3')|...
        (wellLabels_change(:, 1) == 'S') & (wellLabels_change(:, 2) == 'E2')|...
        (wellLabels_change(:, 1) == 'S') & (wellLabels_change(:, 2) == 'E1')|...
        (wellLabels_change(:, 1) == 'S') & (wellLabels_change(:, 2) == 'D');

    anoms = [anoms;washTables(anom_indx, :)];
end



%% Show & Save all anomalous transitions

%Create base import directory
baseDir = 'Preprocessed Data\';

%Create the export directories if they don't exist
dirExport = fullfile(dirExport,'Anomalies');
if ~exist(dirExport, 'dir')
    mkdir(dirExport)
end


%Iterate through the anomalies
for ii = 1:1:height(anoms)
    %Get the cell's initial and final category
    intialCat = anoms.AvgLabelHistory{ii}(1);  %Average lable starts at frame 100
    finalCat = anoms.AvgLabelHistory{ii}(500); % 100+500=600

    %Create a string to define the transformation
    transform = strcat(string(intialCat)," → " ,string(finalCat));

    %Create an import directory from the each cell's data
    dirImages = fullfile(baseDir, anoms.Donor{ii}, anoms.Wash{ii}, num2str(1), 'Images');
    dirWellBboxes = fullfile(baseDir, anoms.Donor{ii}, anoms.Wash{ii}, num2str(1), "Stats/Well_Bboxes.mat");

    %load images
    imds = imageDatastore(dirImages);
    %load well bboxes
    load(dirWellBboxes, 'maxBboxes');

    %Get the bounding boxes of all wells in the first frame
    All_Wells_Bboxes = maxBboxes;

    %Get the well number
    wellNum = str2num(anoms.Well{ii}(6:end));
    %Get the bounding box for the well of interest
    Well_of_Interest = All_Wells_Bboxes(wellNum,:);

    %Get the rows and columns that span the well of interest
    points = bbox2points(Well_of_Interest);
    %Create row and column spans from points
    cols = points(1,1):1:points(2,1)-1;
    rows = points(1,2):1:points(3,2)-1;

    %Set the cell crop size
    cropSize = 31;

    %Create a variable to hold the strip of images
    imageStrip  =[];

    %Extract frames 1, 120, 240, 360, 480, 600 of the cell
    for jj = [10, 30, 140, 360, 480, 600]
        %Get the current frame
        Current_Frame = imread(imds.Files{jj});
        %Extract the well
        Well_Frame0 = Current_Frame(rows,cols);

        %Get the cell of interest's centroids
        centroids = anoms.CentroidHistory{ii}(jj,:);

        %Get the x and y positions of the rectangle based on the centroid and crop size
        xPos = centroids(1)-(cropSize/2);
        yPos = centroids(2)-(cropSize/2);

        %Define the bounding box based on the centroid and resolution
        boundingBox = [xPos, yPos, cropSize, cropSize];

        %Crop the image in a temporary variable
        croppedCell = imcrop(Well_Frame0, boundingBox);

        %Resize the cropped cell to 227X227 & convert to rgb
        croppedCell = gray2rgb(imresize(croppedCell, [227 227]));

        %Create a frame titles
        if jj == 10
            frameTitleX = insertText(gray2rgb(uint8(ones(30, 227))*255),[70, 1], strcat(num2str(jj), "[s]"), 'FontSize', 20, 'BoxColor', 'white' );
            %Concatenate the cropped cell and frame title
            croppedCell2 = cat(1,croppedCell, frameTitleX );
            frameTitleY = insertText(gray2rgb(uint8(ones(257, 227))*255),[6, 30],  anoms.Donor{ii}, 'FontSize', 25, 'BoxColor', 'white' );
            frameTitleY = insertText(frameTitleY,[6, 80],  anoms.Wash{ii}, 'FontSize', 25, 'BoxColor', 'white' );
            frameTitleY = insertText(frameTitleY,[6, 130],  anoms.Well{ii}, 'FontSize', 25, 'BoxColor', 'white' );
            %Concatenate the cropped cell and frame title
            croppedCell2 = cat(2, frameTitleY, croppedCell2 );
        else
            frameTitleX = insertText(gray2rgb(uint8(ones(30, 227))*255),[70, 1], strcat(num2str(jj), "[s]"), 'FontSize', 20, 'BoxColor', 'white' );
            %Concatenate the cropped cell and frame title
            croppedCell2 = cat(1,croppedCell, frameTitleX );
        end


        %Concatenate the cropped cell horizontally
        imageStrip = cat(2, imageStrip,croppedCell2);


    end

    %Add a title on top showing the transformation
    frameTitle = insertText(gray2rgb(uint8(ones(50, 227*7))*255),[227*3, -10], strcat("Possible Anomaly #", num2str(ii), ": ", transform), 'FontSize', 35, 'BoxColor', 'white' );
    %Concatenate the cropped cell vertically with the title
    imageStrip = cat(1, frameTitle, imageStrip);
    %Create a cap for the right of the image strip
    cap = gray2rgb(uint8(ones(307, 30))*255);
    %Concatenate the cropped cell and right cap
    imageStrip = cat(2,imageStrip, cap );
    %Create and figure and display the image
    fig1 = figure();
    imshow(imageStrip);

    %Export the images to the Data Analysis Folder
    imwrite(imageStrip,fullfile(dirExport,strcat("Anom_",num2str(ii),".png")))
    saveas(fig1,fullfile(dirExport,strcat("Anom_",num2str(ii),".m")))
    
end
