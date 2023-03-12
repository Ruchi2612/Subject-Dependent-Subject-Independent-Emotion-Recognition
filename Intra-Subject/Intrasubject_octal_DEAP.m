%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Independent (for DEAP dataset)
% Octal Classification
%--------------------------------------------------------------------------
% 1. Load Dataset
% 2. Octal Label Construction
% 3. Data Preparation Phase
% 4. Train-Validation-Test Data Split 
% 5. Feature Extraction & Classification using DCNN (for valence-arousal-dominance)
%--------------------------------------------------------------------------

clc,clear all,close all
tic
emotions = dir('D:\Ruchilekha\EmotionDataset\DEAP\S');
E = cell(40 , 1 );
data_folder = fullfile('D:\Ruchilekha\EmotionDataset\DEAP\data_preprocessed_matlab');
cd(data_folder);

for i = 3 : 42
    E{i - 2 , 1} = (emotions(i , 1).name);
end
no_subject =32;
verbatim = 0;
FVector = [];
e=1;
%-------------------------Train Data---------------------------------------

for l = 1:no_subject                                                        % Subjects
    mat=[];
    st = num2str(l);
    if l < 10
        name = strcat('S' , '0' , st, '.mat' );
    else
        name = strcat('S' , st, '.mat' );
    end
    cd(data_folder);
    
    % Load Dataset
    data_str = load(name);
    labels = data_str.labels;
    valence = labels(:,1);
    arousal = labels(:,2);
    dominance = labels(:,3);
    
    % Octal Label Construction
    for k = 1:40                                                           % Video/ Trials      
        % Valence-Arousal-Dominance
        if valence(k)<= 4.5 && arousal(k)<= 4.5 && dominance(k)<= 4.5
            label = 0;
        elseif valence(k)<= 4.5 && arousal(k)<= 4.5 && dominance(k)> 4.5
            label = 1;
        elseif valence(k)<= 4.5 && arousal(k)> 4.5 && dominance(k)<= 4.5
            label = 2;
        elseif valence(k)<= 4.5 && arousal(k)> 4.5 && dominance(k)> 4.5
            label = 3;
        elseif valence(k)> 4.5 && arousal(k)<= 4.5 && dominance(k)<= 4.5
            label = 4;
        elseif valence(k)> 4.5 && arousal(k)<= 4.5 && dominance(k)> 4.5
            label = 5;
        elseif valence(k)> 4.5 && arousal(k)> 4.5 && dominance(k)<= 4.5
            label = 6;
        else 
            label = 7;
        end
        
        emotion{k, 1} = E(k , 1);
        for i = 1 : 32                                                     % Channels  (40x32x8064)
            em(k , i, :) = data_str.data(k, i , : );
        end
        
        for j=1:32                                                         % Channels  (32x7680)
            x_mean = mean(em(k,j,1:384));
            x(j,:)= em(k,j,385:8064) - x_mean;
        end
        
        % Data Preparation Phase
        for t = 1:119
            j_=1;
            for j=1:81
                if j==3 || j==7 || j==12 || j==16 || j==19 || j==21 || j==23 || j==25 || j==27 || j==29 || j==31 || j==33 || j==35 || j==37 || j==39 || j==41 || j==43 || j==45 || j==47 || j==49 || j==51 || j==53 || j==55 || j==57 || j==59 || j==61 || j==63 || j==66 || j==70 || j==77 || j==78 || j==79
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
        if k==1
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
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DEAP_intrasubject\Octal')
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
     