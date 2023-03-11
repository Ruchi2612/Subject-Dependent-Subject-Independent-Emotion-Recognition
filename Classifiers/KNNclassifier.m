   function [accuracy,stats,C] = KNNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
   %KNN   
    knn = fitcknn(trainFeatures, YTraining,'NumNeighbors',5);
    predictedLabels = predict(knn, testFeatures);
    accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

    C = confusionmat(YTesting,predictedLabels);
    [stats] = statsOfMeasure(C, verbatim);