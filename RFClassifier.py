import pandas as pd
import time
import numpy as np
import sklearn.ensemble as es
import DatasetManager as dsm
import argparse

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

MEASURES = ['MORT_30_AMI','MORT_30_CABG','MORT_30_COPD','MORT_30_HF','MORT_30_PN','MORT_30_STK','READM_30_AMI','READM_30_CABG','READM_30_COPD','READM_30_HF','READM_30_HIP_KNEE','READM_30_HOSP_WIDE','READM_30_PN','READM_30_STK']

class RFClassifier:

    def __init__(self,num_est,diagnostics = False,csv = False,mm = 'gini'):
        self.num_estimators = num_est
        self.state_rfc = es.RandomForestClassifier(n_estimators = self.num_estimators,criterion=mm)
        self.region_rfc = es.RandomForestClassifier(n_estimators = self.num_estimators,criterion=mm)
        self.show_diagnostics = diagnostics
        self.csv = csv
        self.diagnostic_log("Creating {0}-estimator RFC classifier...".format(self.num_estimators))

    def diagnostic_log(self,string_output):
        if self.show_diagnostics:
            print(string_output)

    def perform_classification(self,train,test):
        self.diagnostic_log('Running simulation for {0} as training set size, {1} as number of estimators.'.format(len(train),self.num_estimators))

        self.state_rfc.fit(train[MEASURES],train['State'])
        self.region_rfc.fit(train[MEASURES],train['Region'])

        test['Predicted State'] = self.state_rfc.predict(test[MEASURES]).tolist()
        test['Predicted Region'] = self.region_rfc.predict(test[MEASURES]).tolist()

        states_results = (test['State']==test['Predicted State']).apply(boolToInt)
        regions_results = (test['Region']==test['Predicted Region']).apply(boolToInt)
        self.diagnostic_log('  States: {0}/{1} = {2}'.format(sum(states_results),len(states_results),float(sum(states_results)/float(len(states_results)))))
        self.diagnostic_log('  Regions: {0}/{1} = {2}'.format(sum(regions_results),len(regions_results),float(sum(regions_results)/float(len(regions_results)))))

        if self.csv:
            self.diagnostic_log("Creating 'tested.csv' file to show output...")
            test.to_csv('tested.csv')
	
	return test
        

def boolToInt(b):
    return int(b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Perform Random Forest Classification')

    parser.add_argument('-t',dest='training_size',type=int,default=700,help='number of rows in the training set')
    parser.add_argument('-n',dest='num_estimators',type=int,default=1000,help='number of estimators for the random forest')
    parser.add_argument('-v','--verbose',action='store_true',help='Generate verbose output')
    parser.add_argument('-csv','--csv',action='store_true',help='Generate csv file at end')

    args = parser.parse_args()

    rfc = RFClassifier(args.num_estimators,diagnostics = args.verbose, csv = args.csv)


    ds_manager = dsm.DatasetManager(args.training_size)
    train,test = ds_manager.getSets()

    rfc.perform_classification(train,test)

