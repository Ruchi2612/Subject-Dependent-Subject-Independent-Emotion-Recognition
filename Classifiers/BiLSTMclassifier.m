   function [accuracy,stats,C] = BiLSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    numObservationsTrain = size(trainFeatures,1);
    numObservationsTest = size(testFeatures,1);
    classes = length(unique(YTraining));
    
    % BiLSTM   
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
    bilstmLayer(numHiddenUnits,OutputMode="last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

    options = trainingOptions("adam", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=1, ...
    MaxEpochs=5, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest", ...
    Shuffle="never", ...
    Verbose=0); %Plots="training-progress"
    
    net = trainNetwork(XTrain, YTraining,layers,options);
    YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest");
    accuracy = sum(YPred == YTesting)./numel(YTesting);
    
    C = confusionmat(YTesting,YPred);
    [stats] = statsOfMeasure(C, verbatim);