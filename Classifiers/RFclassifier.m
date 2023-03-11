    function [accuracy,stats,C] = RFclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
    if len==2
        %RF   
        t = templateTree('Reproducible',true);
        ensemble = fitcensemble(trainFeatures, YTraining,'Method','Bag','Learners',t);
        predictedLabels = predict(ensemble, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

        C = confusionmat(YTesting ,predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    else
        %RF   
        t = templateTree('Reproducible',true);
        ensemble = fitcensemble(trainFeatures, YTraining,'Method','Bag','Learners',t);
        predictedLabels = predict(ensemble, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

        C = confusionmat(YTesting ,predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    end