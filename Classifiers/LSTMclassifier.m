   function [accuracy,stats,C] = LSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    numObservationsTrain = size(trainFeatures,1);
    numObservationsTest = size(testFeatures,1);
    classes = length(unique(YTraining));
    
    % LSTM   
    inputSize = size(trainFeatures,2);
    numHiddenUnits = 10;
    numClasses = classes;
    miniBatchSize = 8;
    
    for i=1:numObservationsTrain             % TainData
        XTrain{i} = trainFeatures(i,:)';
    end
    for i=1:numObservationsTest             % TestData
        XTest{i} = testFeatures(i,:)';
    end

    layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,OutputMode="last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

    options = trainingOptions("adam", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=1, ...
    MaxEpochs=10, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest", ...
    Shuffle="never", ...
    Verbose=0); % Plots="training-progress"

    net = trainNetwork(XTrain, YTraining,layers,options);
    YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize);
    accuracy = sum(YPred == YTesting)./numel(YTesting);
    
    C = confusionmat(YTesting,YPred);
    [stats] = statsOfMeasure(C, verbatim);