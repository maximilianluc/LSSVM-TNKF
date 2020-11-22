
% First remove rows with NAN, inf, "?" from table
adultdata = readtable('adult_data.csv','TreatAsEmpty',{'.','NA','','?'});

%%% Rename the variables:
adultdata.Properties.VariableNames{1}   = 'age';
adultdata.Properties.VariableNames{2}   = 'workclass';
adultdata.Properties.VariableNames{3}   = 'fnlwgt';
adultdata.Properties.VariableNames{4}   = 'education';
adultdata.Properties.VariableNames{5}   = 'educationnum';
adultdata.Properties.VariableNames{6}   = 'maritalstatus';
adultdata.Properties.VariableNames{7}   = 'occupation';
adultdata.Properties.VariableNames{8}   = 'relationship';
adultdata.Properties.VariableNames{9}   = 'race';
adultdata.Properties.VariableNames{10}  = 'sex';
adultdata.Properties.VariableNames{11}  = 'capitalgain';
adultdata.Properties.VariableNames{12}  = 'capitalloss';
adultdata.Properties.VariableNames{13}  = 'hoursperweek';
adultdata.Properties.VariableNames{14}  = 'nativecountry';
adultdata.Properties.VariableNames{15}  = 'labels';

%%% Remove missing values
TF = ismissing(adultdata,{'' '.' 'NA' NaN '?'});
adultdata(any(TF,2),:) = [];

%%% Categorical data
categorical_workingclass    = categorical(adultdata.workclass);
categorical_education       = categorical(adultdata.education);
categorical_maritalstatus   = categorical(adultdata.maritalstatus);
categorical_occupation      = categorical(adultdata.occupation);
categorical_relationship    = categorical(adultdata.relationship);
categorical_race            = categorical(adultdata.race);
categorical_sex             = categorical(adultdata.sex);
categorical_nativecountry   = categorical(adultdata.nativecountry);
categorical_labels          = categorical(adultdata.labels);

order_workingclass  = categories(categorical_workingclass);
order_education     = categories(categorical_education);
order_maritalstatus = categories(categorical_maritalstatus);
order_occupation    = categories(categorical_occupation);
order_relationship  = categories(categorical_relationship);
order_race          = categories(categorical_race);
order_sex           = categories(categorical_sex);
order_nativecountry = categories(categorical_nativecountry);
order_labels        = categories(categorical_labels);

dummy_workingclass  = dummyvar(categorical_workingclass);
dummy_education     = dummyvar(categorical_education);
dummy_maritalstatus = dummyvar(categorical_maritalstatus);
dummy_occupation    = dummyvar(categorical_occupation);
dummy_relationship  = dummyvar(categorical_relationship);
dummy_race          = dummyvar(categorical_race);
dummy_sex           = dummyvar(categorical_sex);
dummy_nativecountry = dummyvar(categorical_nativecountry);
dummy_labels        = dummyvar(categorical_labels);

%%% Create new tables numeric variables + normalization of the numeric data
Data_age            = array2table(normalize(adultdata.age),'VariableNames',cellstr('age'));
Data_fnlwgt         = array2table(normalize(adultdata.fnlwgt),'VariableNames',cellstr('fnlwgt'));
Data_educationnum   = array2table(normalize(adultdata.educationnum),'VariableNames',cellstr('educationnum'));
Data_capitalgain    = array2table(normalize(adultdata.capitalgain),'VariableNames',cellstr('capitalgain'));
Data_capitalloss    = array2table(normalize(adultdata.capitalloss),'VariableNames',cellstr('capitalloss'));
Data_hoursperweek   = array2table(normalize(adultdata.hoursperweek),'VariableNames',cellstr('hoursperweek'));

%%% Create new tables with dummy variables for categorical variables
Data_workingclass  = array2table(dummy_workingclass,'VariableNames',order_workingclass);
Data_education     = array2table(dummy_education,'VariableNames',order_education); 
Data_maritalstatus = array2table(dummy_maritalstatus,'VariableNames',order_maritalstatus);
Data_occupation    = array2table(dummy_occupation,'VariableNames',order_occupation);
Data_relationship  = array2table(dummy_relationship,'VariableNames',order_relationship);
Data_race          = array2table(dummy_race, 'VariableNames', order_race);
Data_sex           = array2table(dummy_sex,'VariableNames',order_sex);
Data_nativecountry = array2table(dummy_nativecountry,'VariableNames',order_nativecountry);
Data_labels = array2table(dummy_labels,'VariableNames',order_labels);

%%% Create [-1 +1] for labels: -1 < 50k a year, +1 > 50k per year
idx_less = dummy_labels(:,1) == 1;
idx_more = dummy_labels(:,1) ~= 1;
idx = idx_less-idx_more;


%%% Design - what variables/featuers to include? 
% I will use all data

X           = [Data_age Data_workingclass Data_fnlwgt ...
              Data_education Data_educationnum Data_maritalstatus...
              Data_occupation Data_relationship Data_sex...
              Data_capitalgain  Data_capitalloss...
              Data_hoursperweek Data_nativecountry]; 

Labels      = Data_labels; 


X           = table2array(X);
Labels      = idx;

