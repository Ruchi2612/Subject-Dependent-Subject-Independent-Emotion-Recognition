   function [accuracy,stats,C] = DTclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
    %KNN   
    dt = fitctree(trainFeatures, YTraining,'MaxNumSplits',5);
    predictedLabels = predict(dt, testFeatures);
    accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

    C = confusionmat(YTesting,predictedLabels);
    [stats] = statsOfMeasure(C, verbatim);