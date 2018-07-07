% run_classifier: This function takes a single record from the
% challenge training set, and uses that record together with the
% accompanying arousal annotations to tune model parameters
% appropriately. You may want to use this function as a basis for your
% own training code.
%
% Written by Mohammad Ghassemi and Benjamin Moody, 2018

function train_classifier(header_file_name)
        % Read record info from the header file
        data = parse_header(header_file_name);

        X_tr = []; Y_tr = [];

        %load all the the data associated with this subject
        signals      = load(data.signal_location); signals = signals.val;
        arousal      = load(data.arousal_location); arousal = arousal.data.arousals;
        fs           = str2num(data.fs);
        n_samples    = str2num(data.n_samples);
        sid          = data.subject_id;
        signal_names = data.signal_names;

        % select the window size and step size we want to use to
        % compute features
        window_size = 300 * fs;
        window_step = 300 * fs;

        % find the index of the SaO2 signal.
        sao2_ind = find(contains(signal_names,'SaO2'));

        % For each 'window', extract the variance of the SaO2
        ind = 1;
        for j = 1:window_step:n_samples-window_step
%                 X_tr(ind) = var(signals(sao2_ind,j:j+window_step));
                Y_tr(ind) = max(arousal(j:j+window_step));
                ind = ind + 1;
        end
%         temporal_ind=find(contains(signal_names, 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2','ABD','CHEST','AIRFLOW','SaO2','ECG'));
        ind2=1;
        fs=60000;
        meaan = [];
    varr=[];
    kurt=[];
    skw=[];
    med=[];
    mod=[];
    maxx=[];
    minn=[];
    
    curvelength = [];
    cl=[];
    
for k = 1:window_step:n_samples-window_step
   
   
        XY= signals(1:13,k:k+window_step);
        m1 = (mean(XY,2))';
        vl = (var(XY,1,2))';
        k1=(kurtosis(XY,0,2))';
        s1=(skewness(XY,0,2))';
        md1=(median(XY,2))';
        mo1=(mode(XY,2))';
        mx1=(max(XY,[],2))';
        mn1=(min(XY,[],2))';
        meaan = [meaan; m1];
        varr = [varr; vl];
        kurt = [kurt; k1];
        skw = [skw; s1];
        med = [med; md1];
        mod = [mod; mo1];
        maxx = [maxx; mx1];
        minn = [minn; mn1];  
        
      
        cl = sum(abs(XY(:,2:end)-XY(:,1:end-1)),2);
        cl = cl';
        curvelength = [curvelength; cl];
        
   
end
    temporal=[meaan,varr,kurt,skw,med,mod,maxx,minn,curvelength];
    % XX =signals(1:13,k:k+window_step);
%         clear cl;
%         cl = sum(abs(XX(:,2:end)-XX(:,1:end-1)),2);
%         cl = cl';
%         curvelength = [curvelength; cl];

% X_tr= [temporal;curvelength];
    
    X_tr=temporal;

        % Set the -1 regions as 1 (treat unscored regions the same as
        % arousals... you may not want to do this.)
        toss = find(Y_tr == -1);
        Y_tr(toss) = 1;

        % Fit a logistic regression for each subject and save their model
        display('Training Model...')
        coeff = glmfit(X_tr,Y_tr','binomial');
        
        % save the model for submission to challenge.i
        display('Saving Model...')
        save([sid '_model'],'coeff');
