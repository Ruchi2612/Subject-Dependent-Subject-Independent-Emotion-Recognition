%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Independent (for Octal Classification)
%--------------------------------------------------------------------------
% 1. Load Feature Vectors obtaied using proposed DCNN
% 2. Octal Classification using varios classifiers (valence/arousal/dominance)
%    (i)AdaBoost, (ii)SVM, (iii)KNN, (iv)DT, (v)RF, (vi)NB, (vii)NN,
%    (viii)LSTM, and (ix)Bi-LSTM
%--------------------------------------------------------------------------


clc, clear all, close all
verbatim = 0;
%--------------------------------------------------------------------------
% VALENCE-AROUSAL-DOMINANCE
%--------------------------------------------------------------------------
    fprintf('VALENCE-AROUSAL-DOMINANCE:\n')                                                                                     
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intrasubject\Octal')
    result_path = fullfile('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intrasubject\Octal\Results');

    XTrain = load(strcat('VAD_TrainFeatures','.mat'));
    YTrain = load(strcat('VAD_TrainLabels','.mat'));
    XTest = load(strcat('VAD_TestFeatures','.mat'));
    YTest = load(strcat('VAD_TestLabels','.mat'));
    
    data = XTrain.trainFeatures;
    label = YTrain.YTraining_VAD;
    s = size(data,1);
    index = randperm(s);
    train = index(1:round(s*0.80));
    test = index(round(s*0.80)+1:end);

    trainFeatures = data(train,:);
    YTraining = label(train);
    testFeatures = data(test,:);
    YTesting = label(test);
    

    %AdaBoost   
    [accuracy,stats,C] = ABclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('AB:%f \n',accuracy); 
    AB_Acc = accuracy;
    AB_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('Val_AB_stats','.xls');
    %writetable(stats,fname);
    AB_CM = C;
    %fname = strcat('Val_AB_confusion','.xls');
    %writematrix(C,fname);
    
    %SVM   
    [accuracy,stats,C] = SVMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('SVM:%f \n',accuracy); 
    SVM_Acc = accuracy;
    SVM_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('SVM_stats','.xls');
    %writetable(stats,fname);
    SVM_CM = C;
    %fname = strcat('SVM_confusion','.xls');
    %writematrix(C,fname);
    
    %KNN   
    [accuracy,stats,C] = KNNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('KNN:%f \n',accuracy); 
    KNN_Acc = accuracy;
    KNN_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('KNN_stats','.xls');
    %writetable(stats,fname);
    KNN_CM = C;
    %fname = strcat('KNN_confusion','.xls');
    %writematrix(C,fname);
    
    %DT
    [accuracy,stats,C] = DTclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('DT:%f \n',accuracy); 
    DT_Acc = accuracy;
    DT_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('DT_stats','.xls');
    %writetable(stats,fname);
    DT_CM = C;
    %fname = strcat('DT_confusion','.xls');
    %writematrix(C,fname);
    
    %Random Forest
    [accuracy,stats,C] = RFclassifier(trainFeatures,YTraining,testFeatures,YTesting, verbatim);
    fprintf('RF:%f \n',accuracy); 
    RF_Acc = accuracy;
    RF_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('RF_stats','.xls');
    %writetable(stats,fname);
    RF_CM = C;
    %fname = strcat('RF_confusion','.xls');
    %writematrix(C,fname);
    
    %NB
    [accuracy,stats,C] = NBclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('NB:%f \n',accuracy); 
    NB_Acc = accuracy;
    NB_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('NB_stats','.xls');
    %writetable(stats,fname);
    NB_CM = C;
    %fname = strcat('NB_confusion','.xls');
    %writematrix(C,fname);
    
    %NN
    [accuracy,stats,C] = NNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('NN:%f \n',accuracy); 
    NN_Acc = accuracy;
    NN_Measures =stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('NN_stats','.xls');
    %writetable(stats,fname);
    NN_CM = C;
    %fname = strcat('NN_confusion','.xls');
    %writematrix(C,fname);
    
    %LSTM
    [accuracy,stats,C] = LSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('LSTM:%f \n',accuracy); 
    LSTM_Acc = accuracy;
    LSTM_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('LSTM_stats','.xls');
    %writetable(stats,fname);
    LSTM_CM = C;
    %fname = strcat('LSTM_confusion','.xls');
    %writematrix(C,fname);
    
    %BiLSTM
    [accuracy,stats,C] = BiLSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('BiLSTM:%f \n',accuracy); 
    BiLSTM_Acc = accuracy;
    BiLSTM_Measures = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('BiLSTM_stats','.xls');
    %writetable(stats,fname);
    BiLSTM_CM = C;
    %fname = strcat('BiLSTM_confusion','.xls');
    %writematrix(C,fname);
    