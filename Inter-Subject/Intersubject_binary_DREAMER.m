%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for DREAMER dataset)
% Binary Classification
%--------------------------------------------------------------------------
% 1. Load Dataset
% 2. Binary Label Construction
% 3. Data Preparation Phase
% 4. Train-Validation-Test Data Split 
% 5. Feature Extraction & Classification using DCNN (for valence)
% 6. Feature Extraction & Classification using DCNN (for arousal)
% 7. Feature Extraction & Classification using DCNN (for dominance)
%--------------------------------------------------------------------------

clc, clear all, close all
data_folder = fullfile('D:\EmotionDataset\');
cd(data_folder);
verbatim = 0;
load DREAMER.mat

for sub=1:23                                                                % 23 participants
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
    
    for trial = 1:18                                                        % 18 videos
        B = baseEeg{trial,1};
        E = eeg{trial,1};
        
        % Binary Label Construction
        if Valence(trial)> 2.5
            label_1 = 1;
        else
            label_1 = 0;
        end
        if Arousal(trial)> 2.5
            label_2 = 1;
        else
            label_2 = 0;
        end
        if Dominance(trial)> 2.5
            label_3 = 1;
        else
            label_3 = 0;
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
            Label_val = lab1';
            Label_aro = lab2';
            Label_dom = lab3';
        else
            Label_val = [Label_val;lab1'];
            Label_aro = [Label_val;lab2'];
            Label_dom = [Label_val;lab3'];
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
    
    YTrain_val = Label_val(idx(1:round(n*0.80))) ; 
    YValid_val = Label_val(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_val = Label_val(idx(round(n*0.90)+1:end)) ;
    
    YTrain_aro = Label_aro(idx(1:round(n*0.80))) ; 
    YValid_aro = Label_aro(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_aro = Label_aro(idx(round(n*0.90)+1:end)) ;
    
    YTrain_dom = Label_dom(idx(1:round(n*0.80))) ; 
    YValid_dom = Label_dom(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_dom = Label_dom(idx(round(n*0.90)+1:end)) ;
    
    XTraining = (XTrain);
    XValidation = (XValid);
    XTesting = (XTest);
    
    YTraining_val = categorical(YTrain_val);
    YValidation_val = categorical(YValid_val);
    YTesting_val = categorical(YTest_val);
    
    YTraining_aro = categorical(YTrain_aro);
    YValidation_aro = categorical(YValid_aro);
    YTesting_aro = categorical(YTest_aro);
    
    YTraining_dom = categorical(YTrain_dom);
    YValidation_dom = categorical(YValid_dom);
    YTesting_dom = categorical(YTest_dom);
    
    %% Extract Features using DCNN for valence
    fprintf('Subject: %d VALENCE :\n',sub)

    cd('D:\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTraining_val));
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_val,XValidation,YValidation_val,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Binary\Valence')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_val');
    %     save(strcat(name,'_TestLabels'),'YTesting_val');
     
    % DCNN classifier
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_val,net,verbatim);
    fprintf('DCNN Accuracy for Valence :%f \n',accuracy);
    DCNN_ValAcc(l,:) = accuracy;
    DCNN_ValMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('Val_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_ValCM(:,:) = C;
    %fname = strcat('Val_DCNN_confusion','.xls');
    %writematrix(C,fname)     
    
    %% Extract Features using DCNN for arousal
    fprintf('Subject: %d   AROUSAL :\n',sub)
    len = length(unique(YTraining_aro));
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_aro,XValidation,YValidation_aro,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Binary\Arousal')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_aro');
    %     save(strcat(name,'_TestLabels'),'YTesting_aro');
    
    % DCNN classifier
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_aro,net,verbatim);
    fprintf('DCNN Accuracy for Arousal :%f \n',accuracy);
    DCNN_AroAcc(l,:) = accuracy;
    DCNN_AroMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('Aro_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_AroCM(:,:) = C;
    %fname = strcat('Aro_DCNN_confusion','.xls');
    %writematrix(C,fname)
    
    %% Extract Features using CNN for dominance
    fprintf('Subject: %d   DOMINANCE :\n',sub)
    len = length(unique(YTraining_dom));
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    load GoogleNet_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_dom,XValidation,YValidation_dom,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DREAMER_intersubject\Binary\Dominance')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %     save(strcat(name,'_TestFeatures'),'testFeatures');
    %     save(strcat(name,'_TrainLabels'),'YTraining_dom');
    %     save(strcat(name,'_TestLabels'),'YTesting_dom');
    
    % DCNN- classifier
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_dom,net,verbatim);
    fprintf('DCNN Accuracy for Dominance :%f \n',accuracy);
    DCNN_DomAcc(l,:) = accuracy;
    DCNN_DomMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %fname = strcat('Dom_DCNN_stats','.xls');
    %writetable(stats,fname);
    DCNN_DomCM(:,:) = C;
    %fname = strcat('Dom_DCNN_confusion','.xls');
    %writematrix(C,fname) 
    
end
