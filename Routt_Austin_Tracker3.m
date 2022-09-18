function [Cells,nextId] = Routt_Austin_Tracker3(Cells,nextId,frameCount, centroids, bboxes, scores, labels)
%This function tracks cells using the 'Cells' structure. Kalman filters are
%used to predict future centroid locations and then make assocations
%between detected centroids and existing cell tracks. Assigned detections
%update their respective cell track, while unassigned detections are given
%new tracks with an ID based on 'nextId'. Existing cell tracks that did not
%have a detection associated with them get their age and consecutive
%invisibity counts increased. Cell tracks are deleted based on their age,
%visibility, and consecutive invisibility counts.



% Inputs:
% Cells = the tracking structure - [-]
% nextID = the ID for the next cell being tracked - [-]
% centroids = the x and y coordinates at the center of each detected cell - [px]
% bboxes = The bounding box of each cell detected cell - [-]


% Outputs:
% Cells = the tracking structure - [-]
% nextID = the ID for the next cell being tracked - [-]

    %Predict the centroid of each cell being tracked and update its bounding box.
    %Loop through cells being tracked
    for j = 1:length(Cells)
        %Get cell j's bounding box
        bbox = Cells(j).bbox;
        
        % Predict the location of cell j's centroid using the cell's Kalman filter.
        predictedCentroid = predict(Cells(j).kalmanFilter);
        
        % Shift the bounding box so that its center is at
        % the predicted location.
        predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
        
        %Store the predicted bounding box
        Cells(j).bbox = [predictedCentroid, bbox(3:4)];
    end
    
    %Assign cell detections in the current frame to existing cells ID'ed
    %via minimizing cost, where cost is the negative log-likelihood of
    %a detection corresponding to an existing cell.
    
    %Find the number of cells being tracked and the number of detected
    %centroids
    nCells = length(Cells);
    nDetections = size(centroids, 1);
    
    %Compute the cost of assigning each detection to each cell already ID'ed.
    
    %Create a matrix to store the cost, M cells by N detections
    cost = zeros(nCells, nDetections);
    %Loop through ID'ed cells and use the distance method to find the
    %cost of assigning every detection
    for j = 1:nCells
        cost(j, :) = distance(Cells(j).kalmanFilter, centroids);
    end
    
    
    %Define cost of not assigning a detection, which is tuned
    %experimentally by the range of values returned by the distance
    %method; too low results in fragmentation of cells and too high
    %yields groups of cells tracked as a single cell
    costOfNonAssignment = 20;
    
    %Solve the assignment problem represented by the cost matrix using
    %the assignDetectionsToTracks function and the costOfNonAssignment
    [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(cost, costOfNonAssignment);
    
    %Update the existing cells ID'ed with their corresponding detections
    numAssignedTracks = size(assignments, 1);
    %Loop through assigned tracks/cells
    for j = 1:numAssignedTracks
        %Define the track/cell ID
        trackIdx = assignments(j, 1);
        %Define the detection ID
        detectionIdx = assignments(j, 2);
        %Define the centroid detected
        centroid = centroids(detectionIdx, :);
        %Define the bounding box detected
        bbox = bboxes(detectionIdx, :);
        %Define the ensemble scores detected
        score = scores(detectionIdx, :);
        %Define the ensemble scores detected
        label = labels(detectionIdx, :);
        
        %Add the centroid and the current frame to the cell's centroid history 
        Cells(trackIdx).frameN = [Cells(trackIdx).frameN; frameCount];
        Cells(trackIdx).centroidHistory = [Cells(trackIdx).centroidHistory; centroid];

        %Add the score and the label to the cell's track 
        Cells(trackIdx).softmaxHistory = [Cells(trackIdx).softmaxHistory; score];
        Cells(trackIdx).labelHistory = [Cells(trackIdx).labelHistory; label];

        %Check if size of softmaxHistory is greater than 100
        windowSize = 100;
        window = [];
        if(height(Cells(trackIdx).softmaxHistory)>=windowSize)
            %Iterate through cropped cells and get softmax for past 10
            %frames and average
            rbcType = categorical({'D','E1','E2','E3','S','SE','ST'});
            for ee = height(Cells(trackIdx).softmaxHistory)-(windowSize-1):1:height(Cells(trackIdx).softmaxHistory)
                window = cat(1, Cells(trackIdx).softmaxHistory{ee}, window);
            end
            movingAvg = mean(window);
            [~, I] = max(movingAvg);
            Cells(trackIdx).avgSoftmaxHistory = [Cells(trackIdx).avgSoftmaxHistory; movingAvg];
            Cells(trackIdx).avgLabelHistory = [Cells(trackIdx).avgLabelHistory; rbcType(I)];
        %If greater than 10, average the last 10 softmax cells
        end
        
        
        % Correct the estimate of the object's location
        % using the new detection's centroid.
        correct(Cells(trackIdx).kalmanFilter, centroid);
        
        % Replace predicted bounding box with the detected bounding box.
        Cells(trackIdx).bbox = bbox;
        
        % Update track's age.
        Cells(trackIdx).age = Cells(trackIdx).age + 1;
        
        % Update visibility count.
        Cells(trackIdx).totalVisibleCount = ...
            Cells(trackIdx).totalVisibleCount + 1;
        
        %Reset invisibility count
        Cells(trackIdx).consecutiveInvisibleCount = 0;
    end
    
    %Update tracks/cells that did not have a detection assigned to them
    %Loop through unassigned tracks/cells
    for  j= 1:length(unassignedTracks)
        %Define the ID of the unassigned track/cell
        ind = unassignedTracks(j);
        %Update the track/cell's age
        Cells(ind).age = Cells(ind).age + 1;
        %Update the track/cell's invisibility count
        Cells(ind).consecutiveInvisibleCount = ...
        Cells(ind).consecutiveInvisibleCount + 1;
    end
    
    %Check if the Cell struct is empty, and, if it isn't, delete lost
    %tracks/cells
    if ~isempty(Cells)
        %Define the limit of frames that cells can be invisible
        invisibleForTooLong = 50;
        %Define an age threshold
        ageThreshold = 8;
        
        % Compute the fraction of the cell's age for which it was visible.
        ages = [Cells(:).age];
        totalVisibleCounts = [Cells(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;   %visibility < 1 is a "blinky" cell
        
        %Define a bad 'blinky' cell as one that is visible less than 60% of its
        %age
        Blinky_Cell_Threshold = 0.3;
        
        
        % Find the indices of 'lost' cells (i.e. young 'blinky' cells or old invisible
        % for too long cells).
        lostInds = (ages < ageThreshold & visibility < Blinky_Cell_Threshold) | ...
            [Cells(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks/cells via their indices.
        Cells = Cells(~lostInds);
    end
    
    
    %Create new tracks/cells from unassigned detections
    
    %Define all detected and unassigned centroids
    centroids = centroids(unassignedDetections, :);
    %Define all detected and unassigned bounding boxes
    bboxes = bboxes(unassignedDetections, :);
    %Define the ensemble scores detected
    if(~isempty(scores(unassignedDetections, :)))
        scores = scores(unassignedDetections, :);
    end

    %Define the ensemble label found
    if(~isempty(labels(unassignedDetections, :)))
        labels = labels(unassignedDetections, :);
    end
    
    
    %Loop through detected and unassigned elements
    for j = 1:size(centroids, 1)
        
        %Define detected and unassigned centroid j
        centroid = centroids(j,:);
        %Define detected and unassigned bounding box j
        bbox = bboxes(j, :);
        %Define detected and unassigned scores j
        score = scores(j,:);
        %Define detected and unassigned label j
        label = labels(j, :);

        % Create a Kalman filter object, assuming a constant velocity model.
        kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
            centroid, [200, 50], [100, 25], 100);
        
        % Create a new track.
        newTrack = struct(...
            'id', nextId, ...
            'bbox', bbox, ...
            'frameN',frameCount, ...  
            'centroidHistory',centroid,...
            'kalmanFilter', kalmanFilter, ...
            'softmaxHistory',{score},...
            'labelHistory',label, ... 
            'avgSoftmaxHistory',[], ...
            'avgLabelHistory', [], ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0);
        
        % Add it to the array of tracks.
        Cells(end + 1) = newTrack;
        
        % Increment the next id.
        nextId = nextId + 1;
    end
 
end


