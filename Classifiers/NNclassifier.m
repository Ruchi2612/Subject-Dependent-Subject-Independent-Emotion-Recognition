   function [accuracy,stats,C] = NNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim)
    len = length(unique(YTraining));
    %NN   
    nn = fitcnet(trainFeatures, YTraining);
    predictedLabels = predict(nn, testFeatures);
    accuracy = sum(predictedLabels == YTesting)/numel(YTesting);

    C = confusionmat(YTesting,predictedLabels);
    [stats] = statsOfMeasure(C, verbatim);
   end

   
% % %    len = length(unique(YTraining));
% % %     %NN   
% % %     p=trainFeatures; % Train Data
% % %     p=p';
% % %    % L={onehotencode(YTraining,2)};
% % % %    for i = 1:length(YTraining) 
% % % %         T{i,:} = onehotencode(YTraining(i),2);    % Target Data (Labels)
% % % %     end
% % %     T=single(YTraining'); 
% % %     net = feedforwardnet(10,'trainlm');
% % %     net = train(net,p,T);
% % %     p2= testFeatures';
% % %     Y2 = net(p2); % Result Labels for Test Data
% % %     %Yp= onehotdecode(Y2,T,1);
% % %     Yp=categorical(Y2)';
% % %     
% % %     accuracy = sum(Yp == YTesting)/numel(YTesting);
% % % 
% % %     C = confusionmat(YTesting,Yp);
% % %     [stats] = statsOfMeasure(C, verbatim);