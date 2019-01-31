import pandas as pd
import time
import numpy as np
import sklearn.ensemble as es
import argparse

#num,cat
#1,A
#2,A
#3,A
#5,A
#6,B
#8,B
#9,B
#10,B
#11,C
#12,C
#13,C
#14,C
#15,C
#16,D
#17,D
#18,D
#19,D

MEASURES = ['num','num2']

class RFClassifier:

    def __init__(self,num_est,diagnostics = False,csv = False):
        self.num_estimators = num_est
        self.rfc = es.RandomForestClassifier(n_estimators = self.num_estimators)
        self.show_diagnostics = diagnostics
        self.csv = csv
        self.diagnostic_log("Creating {0}-estimator RCF classifier...".format(self.num_estimators))

    def diagnostic_log(self,string_output):
        if self.show_diagnostics:
            print(string_output)

    def perform_classification(self,train,test):
        self.diagnostic_log('Running simulation for {0} as training set size, {1} as number of estimators.'.format(len(train),self.num_estimators))

        self.rfc.fit(train[MEASURES],train['cat'])

        self.diagnostic_log(type(test))
        test['predicted cat'] = self.rfc.predict(test[MEASURES]).tolist()

        results = (test['cat']==test['predicted cat']).apply(boolToInt)
        self.diagnostic_log('  Results: {0}/{1} = {2}'.format(sum(results),len(results),float(sum(results)/float(len(results)))))

        if self.csv:
            self.diagnostic_log("Creating 'dummytested.csv' file to show output...")
            test.to_csv('dummytested.csv')
        

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

    dataset = pd.read_csv('dummy.csv')
    setlen = len(dataset.index)
    train = dataset.head(n=args.training_size)
    test = dataset.tail(n=setlen-args.training_size)

    rfc.perform_classification(train,test)

