    function [accuracy,stats,C] = ABclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
    if len==2
        %AdaBoost   
        t = templateTree('Surrogate','On');
        ensemble = fitensemble(trainFeatures, YTraining,'AdaBoostM1',100,t);
        predictedLabels = predict(ensemble, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

        C = confusionmat(YTesting ,predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    else
        %AdaBoost   
        t = templateTree('Surrogate','On');
        ensemble = fitensemble(trainFeatures, YTraining,'AdaBoostM2',100,t);
        predictedLabels = predict(ensemble, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

        C = confusionmat(YTesting ,predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    end