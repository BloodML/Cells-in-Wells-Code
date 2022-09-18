function [name_order,out] = Routt_Austin_Display3(Cells,name_order, Current_Frame_Annotated, Well_Frame, h1, infoPic1)
%Outputs the final image that shows tracking results by taking in Cell
%tracks, the name order, the current annotated frame, and the well frame

% Inputs:
% Cells = the tracking structure - [-]
% name_order = a list of names whose order corresponds to reliable Cell tracks - [-]
% Current_Frame_Annotated = The current frame with the well of interest highlighted - [px^2]
% Well_Frame = the cropped and resized well frame - [px^2]

% Outputs:
%name_order = a list of names whose order corresponds to reliable Cell tracks - [-]
%out = the final output image to be displayed - [px^2]

%Define a minimum visibility count for reliable cell tracks
minVisibleCount = 8;

%Check if the Cell struct is empty
if ~isempty(Cells)
    
    % Only display tracks that have been visible for more than
    % the minimum visibility count. These are reliable tracks.
    reliableTrackInds = ...
        [Cells(:).totalVisibleCount] > minVisibleCount;
    reliableTracks = Cells(reliableTrackInds);
    
    % Display the objects. If an object has not been detected
    % in this frame, display its predicted bounding box.
    if ~isempty(reliableTracks)
        % Get bounding boxes.
        bboxes = cat(1, reliableTracks.bbox);
        
        % Get ids.
        ids = int32([reliableTracks(:).id]);
        
        % Create unique labels for cells and indicate the ones for
        % which we display the predicted rather than the actual
        % location.
        
        
        %Make a list of names
        names = {'Amy', 'Bob', 'Tim', 'Jeff', 'Joe', 'Greg','Francois',...
            'Zac', 'Steve', 'Ahmed', 'Ping','Alex', 'Rin', 'Evelyn',...
            'Carlos', 'Juan', 'Jorge', 'Zoe', 'Dmitry', 'Yuno',...
            'Olga', 'John', 'Fred', 'Harry','Oscar', 'Ash', 'Alice',...
            'Peanut', 'Tessa', 'Emily', 'Kelly', 'Papi' 'Frank', 'Cynthia', ...
            'Ollie', 'Daniel', 'Dana', 'Edward', 'Margaret', 'Amy', 'Latoya', ...
            'Jill', 'Leslie', 'Susan', 'Lisa', 'Brandi', 'Matt', 'Anna',...
            'Ryan', 'Kim', 'Mario', 'Tara', 'Ted', 'Pam', 'Jody', 'Yuki',...
            'Jane', 'James', 'Pat', 'Tom', 'Gustavo', 'Summer', 'Ada', ...
            'Destiny','Eric', 'Sophia', 'Isaac', 'Ivan', 'Duane', 'Helen', ...
            'Mark', 'Gary', 'Jared', 'Van', 'Eva', 'Ida', 'Willie','Asma', 'Sultana'};
        %If the variable 'name_order' is empty, shuffle the list of names;
        %because I can't pick favorites
        if isempty(name_order)
            p = randperm(size(names, 2));
            name_order = names(p);
        end
        %From the shuffled list of names, use the elements that
        %correspond to ids numbers
        labels = name_order(ids).';
        predictedTrackInds = ...
            [reliableTracks(:).consecutiveInvisibleCount] > 0;
        isPredicted = cell(size(labels));
        isPredicted(predictedTrackInds) = {' predicted'};
        labels = strcat(labels, isPredicted);
        morphs = cell(size(labels));
        
        for qq = 1:1:length(ids)
            if (~isempty(reliableTracks(qq).avgLabelHistory))
                 morphs(qq) = strcat(' - ',cellstr(reliableTracks(qq).avgLabelHistory(end)));
            else
                 morphs(qq) = {'-'};
            end
           
        end
        labels = strcat(labels, morphs);

        % Draw the labels on the cells to show tracking.
        Well_Frame = insertObjectAnnotation(Well_Frame, 'rectangle', ...
            bboxes, labels, 'FontSize', 8);
        %Well_Frame = insertMarker(Well_Frame,Cells(1).centroidHistory(end, :));
    end
end


%Combine the annoted original frame and the well of interest frame
out = imtile({Current_Frame_Annotated, Well_Frame, h1, infoPic1},'BorderSize',0);
end

