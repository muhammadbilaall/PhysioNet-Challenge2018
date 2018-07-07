% prepare_submission: This file illustrates how to prepare an entry
% for the PhysioNet/CinC 2018 Challenge.  It first trains a classifier
% for each record in the training set, then runs the classifiers over
% each record in both the training and test sets. The results from the
% training set are used to calculate scores (the average AUROC and
% average AUPRC), and the results from the test set are saved as .vec
% files for submission to PhysioNet.
%
% Written by Mohammad Ghassemi and Benjamin Moody, 2018

% PLEASE NOTE: The script assumes that you have downloaded the data, and is meant
%             to be run from the directory containing the '/training' and '/test'
%             subdirectories

clear all

% STEP 0: Get information on the subject files
[headers_tr, headers_te] = get_file_info;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1: For each of the training subjects, let's build a model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(headers_tr)
        display('--------------------------------------------------')
        display(['Working on Subject ' num2str(i) '/' num2str(length(headers_tr))])
        train_classifier(headers_tr{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 2: Apply the models to the training set, and check performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(headers_tr)
        display('---------------------------------------------------------------')
        display(['Evaluating Models on Training Subject ' num2str(i) '/' num2str(length(headers_tr))])
        predictions = run_classifier(headers_tr{i});

        data = parse_header(headers_tr{i});
        arousal = load(data.arousal_location); arousal = arousal.data.arousals;

        % Compute the Area Under Reciever Operator Curve
        valid = find(arousal ~= -1);
        arousals_valid = arousal(valid);
        pred_valid = predictions(valid);

        % If there are no arousals, skip this subject...
        if length(unique(arousals_valid)) == 1
                display('No arousals detected, skipping subject')
                continue;
        end

        % Evaluate performance on this subject
        [~,~,~,AUC(i)] = perfcurve(arousals_valid, pred_valid, 1);
        [~,~,~,AUCpr(i)] = perfcurve(arousals_valid, pred_valid, 1, ...
                                     'xCrit', 'reca', 'yCrit', 'prec');

        display(['AUROC (so far): ' num2str(mean(AUC)) ' +/- ' num2str(std(AUC))])
        display(['AUPRC (so far): ' num2str(mean(AUCpr)) ' +/- ' num2str(std(AUCpr))])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3: Apply the models to the testing set, and save .vec files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(headers_te)
        display('--------------------------------------------------')
        display(['Scoring Test Subject ' num2str(i) '/' num2str(length(headers_te))])
        predictions = run_classifier(headers_te{i});

        % Save the predictions for submission to the challenge
        display(['Saving predictions'])
        [~, recbase, ~] = fileparts(headers_te{i});
        fileID = fopen([recbase '.vec'], 'w');
        fprintf(fileID, '%.3f\n', predictions);
        fclose(fileID);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 4: Generate a zip file for submission to PhysioNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Delete any files if they existed previously
delete('entry.zip');
% Note: this will not package any sub-directories!
zip('entry.zip', {'*.m', '*.c', '*.mat', '*.vec', '*.txt', '*.sh'});
