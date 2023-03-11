   function [accuracy,stats,C] = NBclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
    %NB   
    nb = fitcnb(trainFeatures, YTraining);
    predictedLabels = predict(nb, testFeatures);
    accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

    C = confusionmat(YTesting,predictedLabels);
    [stats] = statsOfMeasure(C, verbatim);

