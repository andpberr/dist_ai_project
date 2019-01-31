import pandas as pd
import time
import numpy as np
import sklearn.ensemble as es

TRAINING_SIZE = 700
NUM_EST = 1000
MEASURES = ['MORT_30_AMI','MORT_30_CABG','MORT_30_COPD','MORT_30_HF','MORT_30_PN','MORT_30_STK','READM_30_AMI','READM_30_CABG','READM_30_COPD','READM_30_HF','READM_30_HIP_KNEE','READM_30_HOSP_WIDE','READM_30_PN','READM_30_STK']

rcf = es.RandomForestClassifier(n_estimators = NUM_EST )


# Column names:
# 'Provider ID'
# 'Hospital Name'
# 'Address'
# 'City'
# 'State'
# 'ZIP Code'
# 'County Name'
# 'Phone Number'
# 'Measure Name'
# 'Measure ID'
# 'Compared to National'
# 'Denominator'
# 'Score'
# 'Lower Estimate'
# 'Higher Estimate'
# 'Footnote'
# 'Measure Start Date'
# 'Measure End Date'
# 'Location'

dataset = pd.read_csv('Readmissions_and_Deaths_-_Hospital.csv')
#dataset = pd.read_csv('subset.csv')
states = dataset[['Hospital Name','State']]

dataset['Score'].replace("Not Available",None,inplace=True)
dataset['Score'] = dataset['Score'].astype(float)

#pivd = dataset.pivot('Hospital Name','Measure ID','Score')
pivd = pd.pivot_table(dataset,
    values='Score', 
    index='Hospital Name',
    columns='Measure ID',
    aggfunc='mean')


fin = pd.merge(pivd,
    states,
    left_index=True,
    right_on='Hospital Name').drop_duplicates()

fin = fin[~fin['State'].isin(['DC', 'GU', 'MP', 'PR', 'VI'])]

print(len(fin['State'].drop_duplicates()))

print(fin.groupby('State').size())

setlen = len(fin.index)
train = fin.head(n=TRAINING_SIZE)
test = fin.tail(n=setlen-TRAINING_SIZE)

#while len(train['State'].drop_duplicates()) < 49:
    #print(len(train['State'].drop_duplicates()))
    #train = train.reindex(np.random.permutation(train.index))

rcf.fit(train[MEASURES],train['State'])

#print(rcf.predict(test[MEASURES]))
test['Predicted State'] = rcf.predict(test[MEASURES]).tolist()

test.to_csv('tested.csv')
#print(len(test['State'].drop_duplicates()))
#fin['State'].drop_duplicates().to_csv('states.csv')
#train.to_csv('pivoted.csv')

