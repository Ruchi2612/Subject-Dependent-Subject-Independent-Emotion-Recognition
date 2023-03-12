%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for Octal Classification)
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
    YTraining = YTrain.YTraining_VAD;
    testFeatures = XTest.testFeatures;
    YTesting = YTest.YTesting_VAD;
    
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
AB_avg_VAD = mean(AB_Acc);         AB_stas_VAD = mean(AB_Measures,3);           AB_confusion_VAD = sum(AB_CM,3);
SVM_avg_VAD = mean(SVM_Acc);       SVM_stas_VAD = mean(SVM_Measures,3);         SVM_confusion_VAD = sum(SVM_CM,3);
KNN_avg_VAD = mean(KNN_Acc);       KNN_stas_VAD = mean(KNN_Measures,3);         KNN_confusion_VAD = sum(KNN_CM,3);
DT_avg_VAD = mean(DT_Acc);         DT_stas_VAD = mean(DT_Measures,3);           DT_confusion_VAD = sum(DT_CM,3);
RF_avg_VAD = mean(RF_Acc);         RF_stas_VAD = mean(RF_Measures,3);           RF_confusion_VAD = sum(RF_CM,3);
NB_avg_VAD = mean(NB_Acc);         NB_stas_VAD = mean(NB_Measures,3);           NB_confusion_VAD = sum(NB_CM,3);
NN_avg_VAD = mean(NN_Acc);         NN_stas_VAD = mean(NN_Measures,3);           NN_confusion_VAD = sum(NN_CM,3);
LSTM_avg_VAD = mean(LSTM_Acc);     LSTM_stas_VAD = mean(LSTM_Measures,3);       LSTM_confusion_VAD = sum(LSTM_CM,3);
BiLSTM_avg_VAD = mean(BiLSTM_Acc); BiLSTM_stas_VAD = mean(BiLSTM_Measures,3);   BiLSTM_confusion_VAD = sum(BiLSTM_CM,3);
