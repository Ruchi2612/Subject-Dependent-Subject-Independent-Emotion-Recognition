%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Independent (for DREAMER dataset)
% Quad Classification
%--------------------------------------------------------------------------
% 1. Load Dataset
% 2. Quad Label Construction
% 3. Data Preparation Phase
% 4. Train-Validation-Test Data Split 
% 5. Feature Extraction & Classification using DCNN (for dominance-arousal)
% 6. Feature Extraction & Classification using DCNN (for arousal-dominance)
% 7. Feature Extraction & Classification using DCNN (for dominance-dominance)
%--------------------------------------------------------------------------

clc, clear all, close all
data_folder = fullfile('D:\Ruchilekha\EmotionDataset\');
cd(data_folder);

load DREAMER.mat
%--------------------------------------------------------------------------
no_subject =23;
verbatim = 0;
%-------------------------Train Data---------------------------------------
FVector = [];
e=1;
for sub = 1:no_subject                                                       % Subjects
    mat=[];
    
    % Load Dataset
    S = DREAMER.Data{sub};
    baseEeg = S.EEG.baseline;
    eeg = S.EEG.stimuli;
    baseEcg = S.ECG.stimuli;
    ecg = S.ECG.stimuli;
    Valence = S.ScoreValence;
    Arousal = S.ScoreArousal;
    Dominance = S.ScoreDominance;
    
    for trial = 1:18                                                        % 18 videos
        B = baseEeg{trial,1};
        E = eeg{trial,1};
        
        % Quad Label Construction
        % Valence-Arousal
        if Valence(trial)<= 2.5 && Arousal(trial)<= 2.5
            label_1 = 0;
        elseif Valence(trial)<= 2.5 && Arousal(trial)> 2.5
            label_1 = 1;
        elseif Valence(trial)> 2.5 && Arousal(trial)<= 2.5
            label_1 = 2;
        else 
            label_1 = 3;
        end
        % Arousal-Dominance
        if Arousal(trial)<= 2.5 && Dominance(trial)<= 2.5
            label_2 = 0;
        elseif Arousal(trial)<= 2.5 && Dominance(trial)> 2.5
            label_2 = 1;
        elseif Arousal(trial)> 2.5 && Dominance(trial)<= 2.5
            label_2 = 2;
        else 
            label_2 = 3;
        end
        % Dominance-Valence
        if Dominance(trial)<= 2.5 && Valence(trial)<= 2.5
            label_3 = 0;
        elseif Dominance(trial)<= 2.5 && Valence(trial)> 2.5
            label_3 = 1;
        elseif Dominance(trial)> 2.5 && Valence(trial)<= 2.5
            label_3 = 2;
        else 
            label_3 = 3;
        end
        
        for ch = 1:14                                                      % 14 channels
            base = mean(B(:,ch));
            x(ch,:) = E(length(E)-7679:end,ch)-base;
        end
        
        % Data Preparation Phase
        for t = 1:119
            j_=1;
            for j=1:81
                if j==12 || j==16 || j==19 || j==21 || j==25 || j==27 || j==29 || j==35 || j==37 || j==45 || j==55 || j==63 || j==76 || j==78
                    sig(j,:) = (x(j_,64*(t-1)+1:128+ 64*(t-1)));
                    j_=j_+1;
                else
                    sig(j,:)= zeros(1,128); 
                end
            end
                
           org(:,:,1,t) = sig;
           lab1(t)=label_1;
           lab2(t)=label_2;
           lab3(t)=label_3;
        end
        mat = cat(4,mat,org);
        if trial==1
            Label_VA = lab1';
            Label_AD = lab2';
            Label_DV = lab3';
        else
            Label_VA = [Label_VA;lab1'];
            Label_AD = [Label_AD;lab2'];
            Label_DV = [Label_DV;lab3'];
        end
        lab =[];
    end 
    
    if e==1
        FLabel_VA = Label_VA;
        FLabel_AD = Label_AD;
        FLabel_DV = Label_DV;
    else
        FLabel_VA = [FLabel_VA ; Label_VA];
        FLabel_AD = [FLabel_AD ; Label_AD];
        FLabel_DV = [FLabel_DV ; Label_DV];
    end 
    FVector = cat(4,FVector,mat);
    e=e+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Train-Validation-Test Data Split 
    [p,q,m,n] = size(FVector) ; 
    idx = randperm(n) ;  % shuffle the rows 
    XTrain = FVector(:,:,:,idx(1:round(n*0.80))) ;  
    XValid = FVector(:,:,:,idx(round(n*0.80)+1:round(n*0.90))) ;
    XTest = FVector(:,:,:,idx(round(n*0.90)+1:end))  ; 
    
    YTrain_VA = FLabel_VA(idx(1:round(n*0.80))) ; 
    YValid_VA = FLabel_VA(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_VA = FLabel_VA(idx(round(n*0.90)+1:end)) ;

    YTrain_AD = FLabel_AD(idx(1:round(n*0.80))) ; 
    YValid_AD = FLabel_AD(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_AD = FLabel_AD(idx(round(n*0.90)+1:end)) ;

    YTrain_DV = FLabel_DV(idx(1:round(n*0.80))) ; 
    YValid_DV = FLabel_DV(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_DV = FLabel_DV(idx(round(n*0.90)+1:end)) ;

    XTraining = (XTrain);
    XValidation = (XValid);
    XTesting = (XTest);

    YTraining_VA = categorical(YTrain_VA);
    YValidation_VA = categorical(YValid_VA);
    YTraining_AD = categorical(YTrain_AD);
    YValidation_AD = categorical(YValid_AD);
    YTraining_DV = categorical(YTrain_DV);
    YValidation_DV = categorical(YValid_DV);

    YTesting_VA = categorical(YTest_VA);
    YTesting_AD = categorical(YTest_AD);
    YTesting_DV = categorical(YTest_DV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Extract Features using DCNN for valence-arousal
    fprintf('VALENCE-AROUSAL :\n')
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTrain_VA));
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else 
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
    net = TrainNetworkCode(XTraining,YTraining_VA,XValidation,YValidation_VA,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DREAMER_intrasubject\Quad')
    %     save('VAnet.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('VA_TrainFeatures'),'trainFeatures');
    %     save(strcat('VA_TestFeatures'),'testFeatures');
    %     save(strcat('VA_TrainLabels'),'YTraining_VA');
    %     save(strcat('VA_TestLabels'),'YTesting_VA');
     
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_VA,net,verbatim);
     fprintf('DCNN Accuracy for Valence-Arousal :%f \n',accuracy);
     DCNN_ValAro = accuracy;
    %      fname = strcat('VA_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('VA_DCNN_confusion','.xls');
    %      writematrix(C,fname);
    
    %% Extract Features using DCNN for arousal-dominance
    fprintf('AROUSAL-DOMINANCE :\n')
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTrain_AD));
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
  
    net = TrainNetworkCode(XTraining,YTraining_AD,XValidation,YValidation_AD,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DREAMER_intrasubject\Quad')
    %     save('ADnet.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('AD_TrainFeatures'),'trainFeatures');
    %     save(strcat('AD_TestFeatures'),'testFeatures');
    %     save(strcat('AD_TrainLabels'),'YTraining_AD');
    %     save(strcat('AD_TestLabels'),'YTesting_AD');
    
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_AD,net,verbatim);
     fprintf('DCNN Accuracy for Arousal-Dominance :%f \n',accuracy);
     DCNN_AroDom = accuracy;
    %      fname = strcat('AD_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('AD_DCNN_confusion','.xls');
    %      writematrix(C,fname);

    %% Extract Features using DCNN for dominance-valence
    fprintf('DOMINANCE-VALENCE :\n')
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
     len = length(unique(YTrain_DV));
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
    
    net = TrainNetworkCode(XTraining,YTraining_DV,XValidation,YValidation_DV,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DREAMER_intrasubject\Quad')
    %     save('DVnet.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('DV_TrainFeatures'),'trainFeatures');
    %     save(strcat('DV_TestFeatures'),'testFeatures');
    %     save(strcat('DV_TrainLabels'),'YTraining_DV');
    %     save(strcat('DV_TestLabels'),'YTesting_DV');
    
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_DV,net,verbatim);
     fprintf('DCNN Accuracy for Dominance-Valence :%f \n',accuracy);
     DCNN_DomVal = accuracy;
    %      fname = strcat('DV_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('DV_DCNN_confusion','.xls');
    %      writematrix(C,fname);
