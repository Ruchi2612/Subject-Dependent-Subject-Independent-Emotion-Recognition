%--------------------------------------------------------------------------
% Author: Ruchilekha
% Date:   11/03/2023
%--------------------------------------------------------------------------
% Code for Subject Dependent (for DEAP dataset)
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
emotions = dir('D:\EmotionDataset\DEAP\S');
E = cell(40 , 1 );

data_folder = fullfile('D:\EmotionDataset\DEAP\data_preprocessed_matlab');
cd(data_folder);

for i = 3 : 42
    E{i - 2 , 1} = (emotions(i , 1).name);
end
cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha');
%%
FVector = [];
verbatim = 0;
for l = 1:32                                                                % Subjects
    mat=[];
    st = num2str(l);
    if l < 10
        name = strcat('S' , '0' , st );
    else
        name = strcat('S' , st );
    end
    cd(data_folder);
    
    % Load Dataset
    data_str = load(strcat(name,'.mat'));
    labels = data_str.labels;
    valence = labels(:,1);
    arousal = labels(:,2);
    dominance = labels(:,3);
    
    for k = 1:40                                                            % Video/ Trials     
        
        % Binary Label Construction 
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
        for i = 1 : 32                                                      % Channels  (40x32x8064)
            em(k , i, :) = data_str.data(k, i , : );
        end
        for j=1:32                                                          % Channels  (32x7680)
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
    fprintf('Subject: %d   VALENCE :\n',l)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_val,XValidation,YValidation_val,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DEAP_intersubject\Binary\Valence')
    %     save(name,'net');
    
    featureLayer = 'fc_1';
    trainFeatures = activations(net, XTraining, featureLayer, 'OutputAs','rows');
    testFeatures = activations(net, XTesting, featureLayer,'OutputAs', 'rows');
    %      save(strcat(name,'_TrainFeatures'),'trainFeatures');
    %      save(strcat(name,'_TestFeatures'),'testFeatures');
    %      save(strcat(name,'_TrainLabels'),'YTraining_val');
    %      save(strcat(name,'_TestLabels'),'YTesting_val');
     
    % DCNN classifier
    [accuracy,stats,C] = DCNNclassifier(XTesting,YTesting_val,net,verbatim);
    fprintf('DCNN Accuracy for Valence :%f \n',accuracy);
    DCNN_ValAcc(l,:) = accuracy;
    fname = strcat('Val_DCNN_stats','.xls');
    DCNN_ValMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %writetable(stats,fname);
    fname = strcat('Val_DCNN_confusion','.xls');
    DCNN_ValCM(:,:) = C;
    %writematrix(C,fname)     
    fprintf('\n')
    
    %% Extract Features using DCNN for arousal
    fprintf('Subject: %d   AROUSAL :\n',l)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_aro,XValidation,YValidation_aro,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DEAP_intersubject\Binary\Arousal')
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
    fname = strcat('Aro_DCNN_stats','.xls');
    DCNN_AroMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %writetable(stats,fname);
    fname = strcat('Aro_DCNN_confusion','.xls');
    DCNN_AroCM(:,:) = C;
    %writematrix(C,fname)
    fprintf('\n')
    
    %% Extract Features using DCNN for dominance
    fprintf('Subject: %d   DOMINANCE :\n',l)
    cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\NeuralNetworkModels');
    load DCNN_VAD_81x128x1_class2ML.mat;
    net = TrainNetworkCode(XTraining,YTraining_dom,XValidation,YValidation_dom,lgraph_1);
    %     cd('C:\Program Files\R2021a\bin\MATLAB\Ruchilekha\Paper1Modify\DEAP_intersubject\Binary\Dominance')
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
    fname = strcat('Dom_DCNN_stats','.xls');
    DCNN_DomMeasures(:,:) = stats{1:9,["classes","microAVG","macroAVG"]};
    %writetable(stats,fname);
    fname = strcat('Dom_DCNN_confusion','.xls');
    DCNN_DomCM(:,:) = C;
    %writematrix(C,fname)
    fprintf('\n')
end
