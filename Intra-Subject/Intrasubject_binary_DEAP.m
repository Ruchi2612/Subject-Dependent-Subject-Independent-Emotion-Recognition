%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Independent (for DEAP dataset)
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

clc,clear all,close all
tic
emotions = dir('D:\Ruchilekha\EmotionDataset\DEAP\S');
E = cell(40 , 1 );
data_folder = fullfile('D:\Ruchilekha\EmotionDataset\DEAP\data_preprocessed_matlab');
cd(data_folder);

for i = 3 : 42
    E{i - 2 , 1} = (emotions(i , 1).name);
end
no_subject = 32;
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
    
    % Binary Label Construction
    for k = 1:40                                                           % Video/ Trials      
        if valence(k)> 4.5
            label_1 = 1;
        else
            label_1 = 0;
        end
        if arousal(k)> 4.5
            label_2 = 1;
        else
            label_2 = 0;
        end
        if dominance(k)> 4.5
            label_3 = 1;
        else
            label_3 = 0;
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
           lab1(t)=label_1;
           lab2(t)=label_2;
           lab3(t)=label_3;
        end
        mat = cat(4,mat,org);
        if k==1
            Label_val = lab1';
            Label_aro = lab2';
            Label_dom = lab3';
        else
            Label_val = [Label_val;lab1'];
            Label_aro = [Label_aro;lab2'];
            Label_dom = [Label_dom;lab3'];
        end
        lab =[];
    end 
    
    if e==1
        FLabel_val = Label_val;
        FLabel_aro = Label_aro;
        FLabel_dom = Label_dom;
    else
        FLabel_val = [FLabel_val ; Label_val];
        FLabel_aro = [FLabel_aro ; Label_aro];
        FLabel_dom = [FLabel_dom ; Label_dom];
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
    
    YTrain_val = FLabel_val(idx(1:round(n*0.80))) ; 
    YValid_val = FLabel_val(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_val = FLabel_val(idx(round(n*0.90)+1:end)) ;

    YTrain_aro = FLabel_aro(idx(1:round(n*0.80))) ; 
    YValid_aro = FLabel_aro(idx(round(n*0.80)+1:round(n*0.90))) ;
    YTest_aro = FLabel_aro(idx(round(n*0.90)+1:end)) ;

    YTrain_dom = FLabel_dom(idx(1:round(n*0.80))) ; 
    YValid_dom = FLabel_dom(idx(round(n*0.80)+1:round(n*0.90))) ; 
    YTest_dom = FLabel_dom(idx(round(n*0.90)+1:end)) ;

    XTraining = (XTrain);
    XValidation = (XValid);
    XTesting = (XTest);

    YTraining_val = categorical(YTrain_val);
    YValidation_val = categorical(YValid_val);
    YTraining_aro = categorical(YTrain_aro);
    YValidation_aro = categorical(YValid_aro);
    YTraining_dom = categorical(YTrain_dom);
    YValidation_dom = categorical(YValid_dom);   

    YTesting_val = categorical(YTest_val);
    YTesting_aro = categorical(YTest_aro);
    YTesting_dom = categorical(YTest_dom);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Extract Features using DCNN for valence
    fprintf('VALENCE :\n')
    len = length(unique(YTrain_val));
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_val,XValidation,YValidation_val,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DEAP_intrasubject\Binary')
    %     save('Val_net.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('Val_TrainFeatures'),'trainFeatures');
    %     save(strcat('Val_TestFeatures'),'testFeatures');
    %     save(strcat('Val_TrainLabels'),'YTraining_val');
    %     save(strcat('Val_TestLabels'),'YTesting_val');
     
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_val,net,verbatim);
     fprintf('DCNN Accuracy for Valence :%f \n',accuracy);
     DCNN_ValAcc = accuracy;
    %      fname = strcat('Val_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('Val_DCNN_confusion','.xls');
    %      writematrix(C,fname);
    
    %% Extract Features using DCNN for arousal
    fprintf('AROUSAL :\n')
    len = length(unique(YTrain_aro));
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_aro,XValidation,YValidation_aro,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DEAP_intrasubject\Binary')
    %     save('Aro_net.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('Aro_TrainFeatures'),'trainFeatures');
    %     save(strcat('Aro_TestFeatures'),'testFeatures');
    %     save(strcat('Aro_TrainLabels'),'YTraining_aro');
    %     save(strcat('Aro_TestLabels'),'YTesting_aro');
    
     %DCNN
     [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_aro,net,verbatim);
     fprintf('DCNN Accuracy for Arousal :%f \n',accuracy);
     DCNN_AroAcc = accuracy;
    %      fname = strcat('Aro_DCNN','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('Aro_DCNN_confusion','.xls');
    %      writematrix(C,fname);
    
    %% Extract Features using DCNN for dominance
    fprintf('DOMINANCE :\n')
    len = length(unique(YTrain_dom));
    cd('C:\Users\CC PC 11\Documents\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_dom,XValidation,YValidation_dom,lgraph_1);
    %     cd('D:\Ruchilekha\EmotionDataset\Paper1Modify\DEAP_intrasubject\Binary')
    %     save('Dom_net.mat','net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %     save(strcat('Dom_TrainFeatures'),'trainFeatures');
    %     save(strcat('Dom_TestFeatures'),'testFeatures');
    %     save(strcat('Dom_TrainLabels'),'YTraining_dom');
    %     save(strcat('Dom_TestLabels'),'YTesting_dom');
    
     %DCNN
     [accuracy,stats] = DCNNclassifier(XTesting,YTesting_dom,net,verbatim);
     fprintf('DCNN Accuracy for Dominance :%f \n',accuracy);
     DCNN_DomAcc = accuracy;
    %      fname = strcat('Dom_DCNN_stats','.xls');
    %      writetable(stats,fname);
    %      fname = strcat('Dom_DCNN_cofusion','.xls');
    %      writematrix(C,fname);