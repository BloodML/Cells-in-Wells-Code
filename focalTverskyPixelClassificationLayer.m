% A Custom Pixel Classification Layer with Focal Tversky Loss

classdef focalTverskyPixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the focal Tversky loss function for training semantic segmentation networks.
    
    % References
    % Jadon, S. (2020). 
    % A survey of loss functions for semantic segmentation. 
    % Retrieved 1 March 2022, from https://arxiv.org/pdf/2006.14822.pdf.
    % ----------
    % Abraham, N., & Khan, N. (2018).
    % A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation.
    % Retrieved 1 March 2022, from https://arxiv.org/pdf/1810.07842.pdf.
    % ----------

    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    properties
        % Default weighting coefficients for false positives and false negatives
        % Default gamma
        Alpha = 0.5;
        Beta = 0.5;
        Gamma = 0.75;   % Actually 1/Gamma from the paper
    end

    methods

        function layer = focalTverskyPixelClassificationLayer(name, alpha, beta, gamma)
            % layer =  focalTverskyPixelClassificationLayer(name) creates a focal Tversky
            % pixel classification layer with the specified name.

            % Set layer name
            layer.Name = name;

            % Set layer properties
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Gamma = gamma;

            % Set layer description
            layer.Description = 'Focal Tversky loss';
        end

        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Tversky loss between
            % the predictions Y and the training targets T.

            Pcnot = 1-Y;
            Gcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            FP = sum(sum(Y.*Gcnot,1),2);
            FN = sum(sum(Pcnot.*T,1),2);

            numer = TP + layer.Epsilon;
            denom = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;

            % Compute focal Tversky index
            lossTIc = power(1 - (numer./denom), layer.Gamma);

            lossTI = sum(lossTIc,3);

            % Return average Tversky index loss
            N = size(Y,4);
            loss = sum(lossTI)/N;

        end

    end
end