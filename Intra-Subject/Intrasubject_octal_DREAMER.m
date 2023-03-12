%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Independent (for DREAMER dataset)
% Octal Classification
%--------------------------------------------------------------------------
% 1. Load Dataset
% 2. Octal Label Construction
% 3. Data Preparation Phase
% 4. Train-Validation-Test Data Split 
% 5. Feature Extraction & Classification using DCNN (for valence-arousal-dominance)
%--------------------------------------------------------------------------

clc, clear all, close all
data_folder = fullfile('D:\Ruchilekha\EmotionDataset\');
cd(data_folder);
load DREAMER.mat
%--------------------------------------------------------------------------
no_subject = 23;
verbatim = 0;
FVector = [];
e=1;
%-------------------------Train Data---------------------------------------

for sub = 1:no_subject                                                      % Subjects
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
        
        % Octal Label Construction
        % Valence-Arousal-Dominance
        if Valence(trial)<= 2.5 && Arousal(trial)<= 2.5 && Dominance(trial)<= 2.5
            label = 0;
        elseif Valence(trial)<= 2.5 && Arousal(trial)<= 2.5 && Dominance(trial)> 2.5
            label = 1;
        elseif Valence(trial)<= 2.5 && Arousal(trial)> 2.5 && Dominance(trial)<= 2.5
            label = 2;
        elseif Valence(trial)<= 2.5 && Arousal(trial)> 2.5 && Dominance(trial)> 2.5
            label = 3;
        elseif Valence(trial)> 2.5 && Arousal(trial)<= 2.5 && Dominance(trial)<= 2.5
            label = 4;
        elseif Valence(trial)> 2.5 && Arousal(trial)<= 2.5 && Dominance(trial)> 2.5
            label = 5;
        elseif Valence(trial)> 2.5 && Arousal(trial)> 2.5 && Dominance(trial)<= 2.5
            label = 6;
        else 
            label = 7;
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
           lab(t)=label;
        end
        mat = cat(4,mat,org);
        if trial==1
            Label_VAD = lab';
        else
            Label_VAD = [Label_VAD;lab'];
        end
        lab =[];
    end 
    
    if e==1
        FLabel_VAD = Label_VAD;
    else
        FLabel_VAD = [FLabel_VAD ; Label_VAD];
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
    
    YTrain_VAD = FLabel_VAD(idx(1:round(n*0.80))) ; 
    YValid_VAD = FLabel_VAD(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_VAD = FLabel_VAD(idx(round(n*0.90)+1:end)) ;

    XTraining = (XTrain);
    XValidation = (XValid);
    XTesting = (XTest);

    YTraining_VAD = categorical(YTrain_VAD);
    YValidation_VAD = categorical(YValid_VAD);
    YTesting_VAD = categorical(YTest_VAD);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Extract Features using DCNN for valence-arousal-dominance
    fprintf('VALENCE-AROUSAL-DOMINANCE :\n')
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    len = length(unique(YTrain_VAD));
    %----------------------------------------------------------------------
    if len == 2
        load DCNN_VAD_81x128x1_class2ML.mat;
    elseif len == 3
        load DCNN_VAD_81x128x1_class3ML.mat;
    elseif len == 4
        load DCNN_VAD_81x128x1_class4ML.mat;
    elseif len == 5
        load DCNN_VAD_81x128x1_class5ML.mat;
    elseif len == 6
        load DCNN_VAD_81x128x1_class6ML.mat;
    elseif len == 7
        load DCNN_VAD_81x128x1_class7ML.mat;
    else
        load DCNN_VAD_81x128x1_class8ML.mat;
    end
    %----------------------------------------------------------------------
    net = TrainNetworkCode(XTraining,YTraining_VAD,XValidation,YValidation_VAD,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DREAMER_intrasubject\Octal')
    %     save('VAD_net.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('VAD_TrainFeatures'),'trainFeatures');
    %     save(strcat('VAD_TestFeatures'),'testFeatures');
    %     save(strcat('VAD_TrainLabels'),'YTraining_VAD');
    %     save(strcat('VAD_TestLabels'),'YTesting_VAD');
     
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_VAD,net,verbatim);
     fprintf('DCNN:%f \n',accuracy);
     DCNN_ValAroDom = accuracy;
    %      fname = strcat('VAD_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('VAD_DCNN_confusion','.xls');
    %      writematrix(C,fname);
