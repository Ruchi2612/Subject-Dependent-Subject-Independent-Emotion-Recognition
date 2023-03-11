    function [accuracy,stats] = ELmulticlassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim)

    %Ensemble   
    t = templateTree('Surrogate','On');
    ensemble = fitensemble(trainFeatures, YTraining,'AdaBoostM2',100,t);
    predictedLabels = predict(ensemble, testFeatures);
    accuracy = sum(predictedLabels == YTesting)/numel(YTesting);
    
    C = confusionmat(YTesting ,predictedLabels);
    [stats] = statsOfMeasure(C, verbatim);