   function [accuracy,stats,C] = SVMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
   len = length(unique(YTraining)); 
   if len == 2
        %SVM   
        svm = fitcsvm(trainFeatures, YTraining,'KernelFunction','linear');
        predictedLabels = predict(svm, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);
        C = confusionmat(YTesting , predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    else
        %SVM 
        t = templateSVM('Standardize',true,'KernelFunction','linear');
        svm = fitcecoc(trainFeatures, YTraining,'Learners',t);
        predictedLabels = predict(svm, testFeatures);
        accuracy = sum(predictedLabels == YTesting)/numel(YTesting);
        C = confusionmat(YTesting , predictedLabels);
        [stats] = statsOfMeasure(C, verbatim);
    end