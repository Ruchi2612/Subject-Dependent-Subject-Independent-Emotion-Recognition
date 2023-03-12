%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for Binary Classification)
%--------------------------------------------------------------------------
% 1. Load Feature Vectors obtaied using proposed DCNN
% 2. Binary Classification using varios classifiers (valence/arousal/dominance)
%    (i)AdaBoost, (ii)SVM, (iii)KNN, (iv)DT, (v)RF, (vi)NB, (vii)NN,
%    (viii)LSTM, and (ix)Bi-LSTM
%--------------------------------------------------------------------------


clc, clear all, close all
verbatim = 0;
%--------------------------------------------------------------------------
% VALENCE
%--------------------------------------------------------------------------
fprintf('VALENCE:\n')
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
    YTraining = YTrain.YTraining_val;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_val;
    
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
AB_avg_V = mean(AB_Acc);         AB_stas_V = mean(AB_Measures,3);           AB_confusion_V = sum(AB_CM,3);
SVM_avg_V = mean(SVM_Acc);       SVM_stas_V = mean(SVM_Measures,3);         SVM_confusion_V = sum(SVM_CM,3);
KNN_avg_V = mean(KNN_Acc);       KNN_stas_V = mean(KNN_Measures,3);         KNN_confusion_V = sum(KNN_CM,3);
DT_avg_V = mean(DT_Acc);         DT_stas_V = mean(DT_Measures,3);           DT_confusion_V = sum(DT_CM,3);
RF_avg_V = mean(RF_Acc);         RF_stas_V = mean(RF_Measures,3);           RF_confusion_V = sum(RF_CM,3);
NB_avg_V = mean(NB_Acc);         NB_stas_V = mean(NB_Measures,3);           NB_confusion_V = sum(NB_CM,3);
NN_avg_V = mean(NN_Acc);         NN_stas_V = mean(NN_Measures,3);           NN_confusion_V = sum(NN_CM,3);
LSTM_avg_V = mean(LSTM_Acc);     LSTM_stas_V = mean(LSTM_Measures,3);       LSTM_confusion_V = sum(LSTM_CM,3);
BiLSTM_avg_V = mean(BiLSTM_Acc); BiLSTM_stas_V = mean(BiLSTM_Measures,3);   BiLSTM_confusion_V = sum(BiLSTM_CM,3);

%--------------------------------------------------------------------------
%Arousal
%--------------------------------------------------------------------------
fprintf('AROUSAL:\n')
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
    XTrain = load(strcat(name,'_TrainFeatures','.mat'));
    YTrain = load(strcat(name,'_TrainLabels','.mat'));
    XTest = load(strcat(name,'_TestFeatures','.mat'));
    YTest = load(strcat(name,'_TestLabels','.mat'));
    
    trainFeatures = XTrain.trainFeatures;
    YTraining = YTrain.YTraining_aro;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_aro;
    
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
AB_avg_A = mean(AB_Acc);         AB_stas_A = mean(AB_Measures,3);           AB_confusion_A = sum(AB_CM,3);
SVM_avg_A = mean(SVM_Acc);       SVM_stas_A = mean(SVM_Measures,3);         SVM_confusion_A = sum(SVM_CM,3);
KNN_avg_A = mean(KNN_Acc);       KNN_stas_A = mean(KNN_Measures,3);         KNN_confusion_A = sum(KNN_CM,3);
DT_avg_A = mean(DT_Acc);         DT_stas_A = mean(DT_Measures,3);           DT_confusion_A = sum(DT_CM,3);
RF_avg_A = mean(RF_Acc);         RF_stas_A = mean(RF_Measures,3);           RF_confusion_A = sum(RF_CM,3);
NB_avg_A = mean(NB_Acc);         NB_stas_A = mean(NB_Measures,3);           NB_confusion_A = sum(NB_CM,3);
NN_avg_A = mean(NN_Acc);         NN_stas_A = mean(NN_Measures,3);           NN_confusion_A = sum(NN_CM,3);
LSTM_avg_A = mean(LSTM_Acc);     LSTM_stas_A = mean(LSTM_Measures,3);       LSTM_confusion_A = sum(LSTM_CM,3);
BiLSTM_avg_A = mean(BiLSTM_Acc); BiLSTM_stas_A = mean(BiLSTM_Measures,3);   BiLSTM_confusion_A = sum(BiLSTM_CM,3);

%--------------------------------------------------------------------------
%Dominance
%--------------------------------------------------------------------------
fprintf('DOMINANCE:\n')
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
    XTrain = load(strcat(name,'_TrainFeatures','.mat'));
    YTrain = load(strcat(name,'_TrainLabels','.mat'));
    XTest = load(strcat(name,'_TestFeatures','.mat'));
    YTest = load(strcat(name,'_TestLabels','.mat'));
    
    trainFeatures = XTrain.trainFeatures;
    YTraining = YTrain.YTraining_dom;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_dom;
    
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
AB_avg_D = mean(AB_Acc);         AB_stas_D = mean(AB_Measures,3);           AB_confusion_D = sum(AB_CM,3);
SVM_avg_D = mean(SVM_Acc);       SVM_stas_D = mean(SVM_Measures,3);         SVM_confusion_D = sum(SVM_CM,3);
KNN_avg_D = mean(KNN_Acc);       KNN_stas_D = mean(KNN_Measures,3);         KNN_confusion_D = sum(KNN_CM,3);
DT_avg_D = mean(DT_Acc);         DT_stas_D = mean(DT_Measures,3);           DT_confusion_D = sum(DT_CM,3);
RF_avg_D = mean(RF_Acc);         RF_stas_D = mean(RF_Measures,3);           RF_confusion_D = sum(RF_CM,3);
NB_avg_D = mean(NB_Acc);         NB_stas_D = mean(NB_Measures,3);           NB_confusion_D = sum(NB_CM,3);
NN_avg_D = mean(NN_Acc);         NN_stas_D = mean(NN_Measures,3);           NN_confusion_D = sum(NN_CM,3);
LSTM_avg_D = mean(LSTM_Acc);     LSTM_stas_D = mean(LSTM_Measures,3);       LSTM_confusion_D = sum(LSTM_CM,3);
BiLSTM_avg_D = mean(BiLSTM_Acc); BiLSTM_stas_D = mean(BiLSTM_Measures,3);   BiLSTM_confusion_D = sum(BiLSTM_CM,3);
   