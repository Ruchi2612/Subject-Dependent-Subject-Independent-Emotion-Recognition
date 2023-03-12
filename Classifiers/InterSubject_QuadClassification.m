%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for Quad Classification)
%--------------------------------------------------------------------------
% 1. Load Feature Vectors obtaied using proposed DCNN
% 2. Quad Classification using varios classifiers (valence/arousal/dominance)
%    (i)AdaBoost, (ii)SVM, (iii)KNN, (iv)DT, (v)RF, (vi)NB, (vii)NN,
%    (viii)LSTM, and (ix)Bi-LSTM
%--------------------------------------------------------------------------

clc, clear all, close all
verbatim = 0;
%--------------------------------------------------------------------------
% VALENCE-AROUSAL
%--------------------------------------------------------------------------
fprintf('VALENCE-AROUSAL:')
for sub = 1:23
    cd('Add feature data path')
    result_path = fullfile('Add result path');
    st = num2str(sub);
    if sub < 10
        name = strcat('S','0',st);
    else
        name = strcat('S',st);
    end
    load(name);
    fprintf('Subject: %s \n',name)
    XTrain = load(strcat(name,'_TrainFeatures','.mat'));
    YTrain = load(strcat(name,'_TrainLabels','.mat'));
    XTest = load(strcat(name,'_TestFeatures','.mat'));
    YTest = load(strcat(name,'_TestLabels','.mat'));
    
    trainFeatures = XTrain.trainFeatures;
    YTraining = YTrain.YTraining_VA;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_VA;
    
%     DCNN
%     [accuracy,stats,C] = DCNNclassifier(testFeatures,YTesting_val,net,verbatim);
%     fprintf('DCNN:%f \n',accuracy);
%     DCNN_Acc(sub,:) = accuracy;
%     DCNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
%     fname = strcat(name,'_DCNN_stats','.xls');
%     writetable(stats,fname);
%     DCNN_CM(:,:,sub)=C;
%     fname = strcat(name,'_DCNN_confusion','.xls');
%     writematrix(C,fname);
    
    %AdaBoost   
    [accuracy,stats,C] = ABclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('AB:%f \n',accuracy); 
    AB_Acc(sub,:) = accuracy;
    AB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_AB_stats','.xls');
    %writetable(stats,fname);
    AB_CM(:,:,sub)=C;
    %fname = strcat(name,'_AB_confusion','.xls');
    %writematrix(C,fname);
    
    %SVM   
    [accuracy,stats,C] = SVMclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('SVM:%f \n',accuracy); 
    SVM_Acc(sub,:) = accuracy;
    SVM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_SVM_stats','.xls');
    %writetable(stats,fname);
    SVM_CM(:,:,sub)=C;
    %fname = strcat(name,'_SVM_confusion','.xls');
    %writematrix(C,fname);
    
    %KNN   
    [accuracy,stats,C] = KNNclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('KNN:%f \n',accuracy); 
    KNN_Acc(sub,:) = accuracy;
    KNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_KNN_stats','.xls');
    %writetable(stats,fname);
    KNN_CM(:,:,sub)=C;
    %fname = strcat(name,'_KNN_confusion','.xls');
    %writematrix(C,fname);
    
    %DT
    [accuracy,stats,C] = DTclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('DT:%f \n',accuracy); 
    DT_Acc(sub,:) = accuracy;
    DT_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_DT_stats','.xls');
    %writetable(stats,fname);
    DT_CM(:,:,sub)=C;
    %fname = strcat(name,'_DT_confusion','.xls');
    %writematrix(C,fname);
    
    %Random Forest
    [accuracy,stats,C] = RFclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('RF:%f \n',accuracy); 
    RF_Acc(sub,:) = accuracy;
    RF_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_RF_stats','.xls');
    %writetable(stats,fname);
    RF_CM(:,:,sub)=C;
    %fname = strcat(name,'_RF_confusion','.xls');
    %writematrix(C,fname);
    
    %NB
    [accuracy,stats,C] = NBclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('NB:%f \n',accuracy); 
    NB_Acc(sub,:) = accuracy;
    NB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NB_stats','.xls');
    %writetable(stats,fname);
    NB_CM(:,:,sub)=C;
    %fname = strcat(name,'_NB_confusion','.xls');
    %writematrix(C,fname);
    
    %NN
    [accuracy,stats,C] = NNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('NN:%f \n',accuracy); 
    NN_Acc(sub,:) = accuracy;
    NN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NN_stats','.xls');
    %writetable(stats,fname);
    NN_CM(:,:,sub)=C;
    %fname = strcat(name,'_NN_confusion','.xls');
    %writematrix(C,fname);
    
    %LSTM
    [accuracy,stats,C] = LSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('LSTM:%f \n',accuracy); 
    LSTM_Acc(sub,:) = accuracy;
    LSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_LSTM_stats','.xls');
    %writetable(stats,fname);
    LSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_LSTM_confusion','.xls');
    %writematrix(C,fname);
    
    %BiLSTM
    [accuracy,stats,C] = BiLSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('BiLSTM:%f \n',accuracy); 
    BiLSTM_Acc(sub,:) = accuracy;
    BiLSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_BiLSTM_stats','.xls');
    %writetable(stats,fname);
    BiLSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_BiLSTM_confusion','.xls');
    %writematrix(C,fname);
    fprintf('\n')
end
AB_avg_VA = mean(AB_Acc);         AB_stas_VA = mean(AB_Measures,3);           AB_confusion_VA = sum(AB_CM,3);
SVM_avg_VA = mean(SVM_Acc);       SVM_stas_VA = mean(SVM_Measures,3);         SVM_confusion_VA = sum(SVM_CM,3);
KNN_avg_VA = mean(KNN_Acc);       KNN_stas_VA = mean(KNN_Measures,3);         KNN_confusion_VA = sum(KNN_CM,3);
DT_avg_VA = mean(DT_Acc);         DT_stas_VA = mean(DT_Measures,3);           DT_confusion_VA = sum(DT_CM,3);
RF_avg_VA = mean(RF_Acc);         RF_stas_VA = mean(RF_Measures,3);           RF_confusion_VA = sum(RF_CM,3);
NB_avg_VA = mean(NB_Acc);         NB_stas_VA = mean(NB_Measures,3);           NB_confusion_VA = sum(NB_CM,3);
NN_avg_VA = mean(NN_Acc);         NN_stas_VA = mean(NN_Measures,3);           NN_confusion_VA = sum(NN_CM,3);
LSTM_avg_VA = mean(LSTM_Acc);     LSTM_stas_VA = mean(LSTM_Measures,3);       LSTM_confusion_VA = sum(LSTM_CM,3);
BiLSTM_avg_VA = mean(BiLSTM_Acc); BiLSTM_stas_VA = mean(BiLSTM_Measures,3);   BiLSTM_confusion_VA = sum(BiLSTM_CM,3);

%--------------------------------------------------------------------------
% AROUSAL-DOMINANCE
%--------------------------------------------------------------------------
fprintf('AROUSAL-DOMINANCE:\n')
for sub = 1:23
    cd('Add feature data path')
    result_path = fullfile('Add result path');
    st = num2str(sub);
    if sub < 10
        name = strcat('S','0',st);
    else
        name = strcat('S',st);
    end
    load(name);
    fprintf('Subject: %s \n',name)
    XTrain = load(strcat(name,'_TrainFeatures','.mat'));
    YTrain = load(strcat(name,'_TrainLabels','.mat'));
    XTest = load(strcat(name,'_TestFeatures','.mat'));
    YTest = load(strcat(name,'_TestLabels','.mat'));
    
    trainFeatures = XTrain.trainFeatures;
    YTraining = YTrain.YTraining_AD;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_AD;
    
%     DCNN
%     [accuracy,stats,C] = DCNNclassifier(testFeatures,YTesting_val,net,verbatim);
%     fprintf('DCNN:%f \n',accuracy);
%     DCNN_Acc(sub,:) = accuracy;
%     DCNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
%     fname = strcat(name,'_DCNN_stats','.xls');
%     writetable(stats,fname);
%     DCNN_CM(:,:,sub)=C;
%     fname = strcat(name,'_DCNN_confusion','.xls');
%     writematrix(C,fname);
    
    %AdaBoost   
    [accuracy,stats,C] = ABclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('AB:%f \n',accuracy); 
    AB_Acc(sub,:) = accuracy;
    AB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_AB_stats','.xls');
    %writetable(stats,fname);
    AB_CM(:,:,sub)=C;
    %fname = strcat(name,'_AB_confusion','.xls');
    %writematrix(C,fname);
    
    %SVM   
    [accuracy,stats,C] = SVMclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('SVM:%f \n',accuracy); 
    SVM_Acc(sub,:) = accuracy;
    SVM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_SVM_stats','.xls');
    %writetable(stats,fname);
    SVM_CM(:,:,sub)=C;
    %fname = strcat(name,'_SVM_confusion','.xls');
    %writematrix(C,fname);
    
    %KNN   
    [accuracy,stats,C] = KNNclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('KNN:%f \n',accuracy); 
    KNN_Acc(sub,:) = accuracy;
    KNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_KNN_stats','.xls');
    %writetable(stats,fname);
    KNN_CM(:,:,sub)=C;
    %fname = strcat(name,'_KNN_confusion','.xls');
    %writematrix(C,fname);
    
    %DT
    [accuracy,stats,C] = DTclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('DT:%f \n',accuracy); 
    DT_Acc(sub,:) = accuracy;
    DT_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_DT_stats','.xls');
    %writetable(stats,fname);
    DT_CM(:,:,sub)=C;
    %fname = strcat(name,'_DT_confusion','.xls');
    %writematrix(C,fname);
    
    %Random Forest
    [accuracy,stats,C] = RFclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('RF:%f \n',accuracy); 
    RF_Acc(sub,:) = accuracy;
    RF_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_RF_stats','.xls');
    %writetable(stats,fname);
    RF_CM(:,:,sub)=C;
    %fname = strcat(name,'_RF_confusion','.xls');
    %writematrix(C,fname);
    
    %NB
    [accuracy,stats,C] = NBclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('NB:%f \n',accuracy); 
    NB_Acc(sub,:) = accuracy;
    NB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NB_stats','.xls');
    %writetable(stats,fname);
    NB_CM(:,:,sub)=C;
    %fname = strcat(name,'_NB_confusion','.xls');
    %writematrix(C,fname);
    
    %NN
    [accuracy,stats,C] = NNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('NN:%f \n',accuracy); 
    NN_Acc(sub,:) = accuracy;
    NN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NN_stats','.xls');
    %writetable(stats,fname);
    NN_CM(:,:,sub)=C;
    %fname = strcat(name,'_NN_confusion','.xls');
    %writematrix(C,fname);
    
    %LSTM
    [accuracy,stats,C] = LSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('LSTM:%f \n',accuracy); 
    LSTM_Acc(sub,:) = accuracy;
    LSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_LSTM_stats','.xls');
    %writetable(stats,fname);
    LSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_LSTM_confusion','.xls');
    %writematrix(C,fname);
    
    %BiLSTM
    [accuracy,stats,C] = BiLSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('BiLSTM:%f \n',accuracy); 
    BiLSTM_Acc(sub,:) = accuracy;
    BiLSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_BiLSTM_stats','.xls');
    %writetable(stats,fname);
    BiLSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_BiLSTM_confusion','.xls');
    %writematrix(C,fname);
    fprintf('\n')
end
AB_avg_AD = mean(AB_Acc);         AB_stas_AD = mean(AB_Measures,3);           AB_confusion_AD = sum(AB_CM,3);
SVM_avg_AD = mean(SVM_Acc);       SVM_stas_AD = mean(SVM_Measures,3);         SVM_confusion_AD = sum(SVM_CM,3);
KNN_avg_AD = mean(KNN_Acc);       KNN_stas_AD = mean(KNN_Measures,3);         KNN_confusion_AD = sum(KNN_CM,3);
DT_avg_AD = mean(DT_Acc);         DT_stas_AD = mean(DT_Measures,3);           DT_confusion_AD = sum(DT_CM,3);
RF_avg_AD = mean(RF_Acc);         RF_stas_AD = mean(RF_Measures,3);           RF_confusion_AD = sum(RF_CM,3);
NB_avg_AD = mean(NB_Acc);         NB_stas_AD = mean(NB_Measures,3);           NB_confusion_AD = sum(NB_CM,3);
NN_avg_AD = mean(NN_Acc);         NN_stas_AD = mean(NN_Measures,3);           NN_confusion_AD = sum(NN_CM,3);
LSTM_avg_AD = mean(LSTM_Acc);     LSTM_stas_AD = mean(LSTM_Measures,3);       LSTM_confusion_AD = sum(LSTM_CM,3);
BiLSTM_avg_AD = mean(BiLSTM_Acc); BiLSTM_stas_AD = mean(BiLSTM_Measures,3);   BiLSTM_confusion_AD = sum(BiLSTM_CM,3);

%--------------------------------------------------------------------------
% DOMINANCE-VALENCE
%--------------------------------------------------------------------------
fprintf('DOMINANCE-VALENCE:')
for sub = 1:23
    cd('Add feature data path')
    result_path = fullfile('Add result path');
    st = num2str(sub);
    if sub < 10
        name = strcat('S','0',st);
    else
        name = strcat('S',st);
    end
    load(name);
    fprintf('Subject: %s \n',name)
    XTrain = load(strcat(name,'_TrainFeatures','.mat'));
    YTrain = load(strcat(name,'_TrainLabels','.mat'));
    XTest = load(strcat(name,'_TestFeatures','.mat'));
    YTest = load(strcat(name,'_TestLabels','.mat'));
    
    trainFeatures = XTrain.trainFeatures;
    YTraining = YTrain.YTraining_DV;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_DV;
    
%     DCNN
%     [accuracy,stats,C] = DCNNclassifier(testFeatures,YTesting_val,net,verbatim);
%     fprintf('DCNN:%f \n',accuracy);
%     DCNN_Acc(sub,:) = accuracy;
%     DCNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
%     fname = strcat(name,'_DCNN_stats','.xls');
%     writetable(stats,fname);
%     DCNN_CM(:,:,sub)=C;
%     fname = strcat(name,'_DCNN_confusion','.xls');
%     writematrix(C,fname);
    
    %AdaBoost   
    [accuracy,stats,C] = ABclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('AB:%f \n',accuracy); 
    AB_Acc(sub,:) = accuracy;
    AB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_AB_stats','.xls');
    %writetable(stats,fname);
    AB_CM(:,:,sub)=C;
    %fname = strcat(name,'_AB_confusion','.xls');
    %writematrix(C,fname);
    
    %SVM   
    [accuracy,stats,C] = SVMclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('SVM:%f \n',accuracy); 
    SVM_Acc(sub,:) = accuracy;
    SVM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_SVM_stats','.xls');
    %writetable(stats,fname);
    SVM_CM(:,:,sub)=C;
    %fname = strcat(name,'_SVM_confusion','.xls');
    %writematrix(C,fname);
    
    %KNN   
    [accuracy,stats,C] = KNNclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('KNN:%f \n',accuracy); 
    KNN_Acc(sub,:) = accuracy;
    KNN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_KNN_stats','.xls');
    %writetable(stats,fname);
    KNN_CM(:,:,sub)=C;
    %fname = strcat(name,'_KNN_confusion','.xls');
    %writematrix(C,fname);
    
    %DT
    [accuracy,stats,C] = DTclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('DT:%f \n',accuracy); 
    DT_Acc(sub,:) = accuracy;
    DT_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_DT_stats','.xls');
    %writetable(stats,fname);
    DT_CM(:,:,sub)=C;
    %fname = strcat(name,'_DT_confusion','.xls');
    %writematrix(C,fname);
    
    %Random Forest
    [accuracy,stats,C] = RFclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('RF:%f \n',accuracy); 
    RF_Acc(sub,:) = accuracy;
    RF_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_RF_stats','.xls');
    %writetable(stats,fname);
    RF_CM(:,:,sub)=C;
    %fname = strcat(name,'_RF_confusion','.xls');
    %writematrix(C,fname);
    
    %NB
    [accuracy,stats,C] = NBclassifier(trainFeatures,YTraining,testFeatures,YTesting,net,verbatim);
    fprintf('NB:%f \n',accuracy); 
    NB_Acc(sub,:) = accuracy;
    NB_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NB_stats','.xls');
    %writetable(stats,fname);
    NB_CM(:,:,sub)=C;
    %fname = strcat(name,'_NB_confusion','.xls');
    %writematrix(C,fname);
    
    %NN
    [accuracy,stats,C] = NNclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('NN:%f \n',accuracy); 
    NN_Acc(sub,:) = accuracy;
    NN_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_NN_stats','.xls');
    %writetable(stats,fname);
    NN_CM(:,:,sub)=C;
    %fname = strcat(name,'_NN_confusion','.xls');
    %writematrix(C,fname);
    
    %LSTM
    [accuracy,stats,C] = LSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('LSTM:%f \n',accuracy); 
    LSTM_Acc(sub,:) = accuracy;
    LSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_LSTM_stats','.xls');
    %writetable(stats,fname);
    LSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_LSTM_confusion','.xls');
    %writematrix(C,fname);
    
    %BiLSTM
    [accuracy,stats,C] = BiLSTMclassifier(trainFeatures,YTraining,testFeatures,YTesting,verbatim);
    fprintf('BiLSTM:%f \n',accuracy); 
    BiLSTM_Acc(sub,:) = accuracy;
    BiLSTM_Measures(:,:,sub)=stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat(name,'_BiLSTM_stats','.xls');
    %writetable(stats,fname);
    BiLSTM_CM(:,:,sub)=C;
    %fname = strcat(name,'_BiLSTM_confusion','.xls');
    %writematrix(C,fname);
    fprintf('\n')
end
AB_avg_DV = mean(AB_Acc);         AB_stas_DV = mean(AB_Measures,3);           AB_confusion_DV = sum(AB_CM,3);
SVM_avg_DV = mean(SVM_Acc);       SVM_stas_DV = mean(SVM_Measures,3);         SVM_confusion_DV = sum(SVM_CM,3);
KNN_avg_DV = mean(KNN_Acc);       KNN_stas_DV = mean(KNN_Measures,3);         KNN_confusion_DV = sum(KNN_CM,3);
DT_avg_DV = mean(DT_Acc);         DT_stas_DV = mean(DT_Measures,3);           DT_confusion_DV = sum(DT_CM,3);
RF_avg_DV = mean(RF_Acc);         RF_stas_DV = mean(RF_Measures,3);           RF_confusion_DV = sum(RF_CM,3);
NB_avg_DV = mean(NB_Acc);         NB_stas_DV = mean(NB_Measures,3);           NB_confusion_DV = sum(NB_CM,3);
NN_avg_DV = mean(NN_Acc);         NN_stas_DV = mean(NN_Measures,3);           NN_confusion_DV = sum(NN_CM,3);
LSTM_avg_DV = mean(LSTM_Acc);     LSTM_stas_DV = mean(LSTM_Measures,3);       LSTM_confusion_DV = sum(LSTM_CM,3);
BiLSTM_avg_DV = mean(BiLSTM_Acc); BiLSTM_stas_DV = mean(BiLSTM_Measures,3);   BiLSTM_confusion_DV = sum(BiLSTM_CM,3);


 