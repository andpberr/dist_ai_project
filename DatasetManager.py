import numpy as np
import pandas as pd

class DatasetManager:
    def __init__(self, training_size):
        self.training_size = training_size

    def getSets(self):
        # Create initial pandas dataframe from full csv file
        dataset = pd.read_csv('Readmissions_and_Deaths_-_Hospital.csv')

        # Create Hospital/State Dataset to merge with flattened metrics dataset later
        states = dataset[['Hospital Name','State']]

        # Pull State -> Region mapping from homemade csv file
        stateregions = pd.read_csv('states.csv')

        # Replace text values of "Not Available" with None (sort of the python equivalent of SQL null)
        dataset['Score'].replace("Not Available",None,inplace=True)
        dataset['Score'] = dataset['Score'].astype(float)

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

        fin = pd.merge(fin,
            stateregions,
            on='State')

	# Perform random shuffle of the rows
	fin = fin.reindex(np.random.permutation(fin.index))

        #rfc.diagnostic_log(fin.groupby('State').size())

        setlen = len(fin.index)
        train = fin.head(n=self.training_size)
        test = fin.tail(n=setlen-self.training_size)
        return [train, test]
