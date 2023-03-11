function [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting,net,verbatim)
    
    %DCNN
%    YPred = predict(net,XTesting);
%     accuracy = mean(YPred == YTesting);
      for T=1:length(YTesting)
         Test=XTesting(:,:,1,T);
         YPredTest(T) = classify(net,Test); 
     end
     accuracy = sum(YPredTest == YTesting')/numel(YTesting);
    
    C = confusionmat(YTesting',YPred);
    [stats] = statsOfMeasure(C, verbatim);
    