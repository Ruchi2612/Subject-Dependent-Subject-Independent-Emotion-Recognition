%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for DREAMER dataset)
% Binary Classification
%--------------------------------------------------------------------------
% 1. Load Dataset
% 2. Quad Label Construction
% 3. Data Preparation Phase
% 4. Train-Validation-Test Data Split 
% 5. Feature Extraction & Classification using DCNN (for valence-arousal)
% 6. Feature Extraction & Classification using DCNN (for arousal-dominance)
% 7. Feature Extraction & Classification using DCNN (for dominance-valence)
%--------------------------------------------------------------------------

clc, clear all, close all
data_folder = fullfile('D:\EmotionDataset\');
cd(data_folder);
verbatim = 0;
load DREAMER.mat

%--------------------------------------------------------------------------
FVector = [];
for sub = 1 : 23                                                             % Subjects
    mat=[];
    st = num2str(sub);
    if sub < 10
        name = strcat('S' , '0' , st );
    else
        name = strcat('S' , st );
    end
    
    % Load Dataset
    S = DREAMER.Data{sub};
    baseEeg = S.EEG.baseline;
    eeg = S.EEG.stimuli;
    baseEcg = S.ECG.stimuli;
    ecg = S.ECG.stimuli;
    Valence = S.ScoreValence;
    Arousal = S.ScoreArousal;
    Dominance = S.ScoreDominance;
    
    for trial = 1:18                                                       % 18 videos
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
    %----------------------------------------------------------------------
    % Train-Validation-Test Data Split (randomly)
    [p,q,m,n] = size(mat) ; 
    idx = randperm(n) ;  % shuffle the rows 
    XTrain = mat(:,:,:,idx(1:round(n*0.80))) ;  
    XValid = mat(:,:,:,idx(round(n*0.80)+1:round(n*0.90))) ;
    XTest = mat(:,:,:,idx(round(n*0.90)+1:end)) ;
    
    YTrain_VA = Label_VA(idx(1:round(n*0.80))) ; 
    YValid_VA = Label_VA(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_VA = Label_VA(idx(round(n*0.90)+1:end)) ;
    
    YTrain_AD = Label_AD(idx(1:round(n*0.80))) ; 
    YValid_AD = Label_AD(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_AD = Label_AD(idx(round(n*0.90)+1:end)) ;
    
    YTrain_DV = Label_DV(idx(1:round(n*0.80))) ; 
    YValid_DV = Label_DV(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_DV = Label_DV(idx(round(n*0.90)+1:end)) ;
    
    XTraining = (XTrain);
    XValidation = (XValid);
    XTesting = (XTest);
    
    YTraining_VA = categorical(YTrain_VA);
    YValidation_VA = categorical(YValid_VA);
    YTesting_VA = categorical(YTest_VA);
    
    YTraining_AD = categorical(YTrain_AD);
    YValidation_AD = categorical(YValid_AD);
    YTesting_AD = categorical(YTest_AD);
    
    YTraining_DV = categorical(YTrain_DV);
    YValidation_DV = categorical(YValid_DV);
    YTesting_DV = categorical(YTest_DV);
    
    %% Extract Features using DCNN for valence-arousal
    fprintf('Subject: %d   VALENCE-AROUSAL :\n',sub)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTraining_VA));
    %----------------------------------------------------------------------
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else 
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
    %----------------------------------------------------------------------
    net = TrainNetworkCode(XTraining,YTraining_VA,XValidation,YValidation_VA,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Quad\Valence_Arousal')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_VA');
    %     save(strcat(name,'_TestLabels'),'YTesting_VA');
     
    %DCNN
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_VA,net,verbatim);
    fprintf('DCNN Accuracy for Valence-Arousal :%f \n',accuracy);
    DCNN_ValAro(l,:) = accuracy;
    DCNN_ValAroMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('ValAro_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_ValAroCM(:,:) = C;
    %fname = strcat('ValAro_DCNN_confusion','.xls');
    %writematrix(C,fname)   
    
    %% Extract Features using DCNN for arousal-dominance
    fprintf('Subject: %d   AROUSAL-DOMINANCE :\n',sub)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTraining_AD));
    %----------------------------------------------------------------------
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
    %----------------------------------------------------------------------
    net = TrainNetworkCode(XTraining,YTraining_AD,XValidation,YValidation_AD,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Quad\Arousal_Dominance')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_AD');
    %     save(strcat(name,'_TestLabels'),'YTesting_AD');
    
    %DCNN
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_AD,net,verbatim);
    fprintf('DCNN Accuracy for Arousal-Dominance :%f \n',accuracy);
    DCNN_AroDom(l,:) = accuracy;
    DCNN_AroDomMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('AroDom_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_AroDomCM(:,:) = C;
    %fname = strcat('AroDom_DCNN_confusion','.xls');
    %writematrix(C,fname) 
    
    %% Extract Features using DCNN for dominance-valence
    fprintf('Subject: %d   DOMINANCE-VALENCE :\n',sub)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTraining_DV));
    %----------------------------------------------------------------------
    if len == 2
         load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
         load DCNN_VAD_81x128x1_class3ML.mat;
    else
         load DCNN_VAD_81x128x1_class4ML.mat;
    end
    %----------------------------------------------------------------------
    net = TrainNetworkCode(XTraining,YTraining_DV,XValidation,YValidation_DV,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Quad\Dominance_Valence')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_DV');
    %     save(strcat(name,'_TestLabels'),'YTesting_DV');
    
    %DCNN
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_DV,net,verbatim);
    fprintf('DCNN Accuracy for Dominance-Valence :%f \n',accuracy);
    DCNN_DomVal(l,:) = accuracy;
    DCNN_DomValMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('DomVal_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_DomValCM(:,:) = C;
    %fname = strcat('DomVal_DCNN_confusion','.xls');
    %writematrix(C,fname)
end
