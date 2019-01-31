import spade
import dummyclassifier as dc
import random as rd

class MyAgent(spade.Agent.Agent):
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def _setup(self):
        self.classifier = dc.RFClassifier(rd.randrange(1000,2000),True,False)

    class ClassifyBehavior(spade.Behaviour.OneShotBehaviour):
        def _process(self):
            # classify
            self.classifier.perform_classification(train,test)

    class InformBehavior(spade.Behaviour.OneShotBehaviour):
        def _process(self):
            # send results
            print("I'm gonna inform!")
