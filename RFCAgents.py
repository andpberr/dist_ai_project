#print("in RFCA, importing RFClassifier...")
import RFClassifier as rfc
#print("in RFCA, importing DatasetManager...")
import DatasetManager as dsm
#print("in RFCA, importing random...")
import random as rd
#print("in RFCA, importing spade...")
import spade
#print("in RFCA, importing pandas...")
import pandas as pd
import os

#TRAINING_SIZE = (rd.randrange(4) + 4) * 100
TRAINING_SIZE = 4445
DS_MANAGER = dsm.DatasetManager(TRAINING_SIZE)
TRAIN,TEST = DS_MANAGER.getSets()
MSG_WAIT = 2

# Each agent will perform its own classification, so they will all have their own instance
# of RFClassifier. Agents differ only in how they communicate
class BaseAgent(spade.Agent.Agent):
	def __init__(self,myID,myPass,mm,nt,nb,bs):
		# Approx 100 trees
		self.classifier = rfc.RFClassifier(rd.randrange(nb+1)*bs + (nt - (nb/2)*bs),True,False,mm)

		# Exactly 1 tree
		#self.classifier = rfc.RFClassifier(1,True,False)


		super(BaseAgent,self).__init__(myID,myPass)

	def getRandom(self,row):
		return rd.random()

	# Find the appropriate list index (over probability list l)
	# for the value c
	def determineIndex(self,l,c):
		if (c > 1):
			return -1 # Error, value must be 0 <= c <= 1
		counter = 1
		while counter <= len(l):
			if sum(l[:counter]) >= c:
				return counter-1
			counter+=1
		
	def chooseStateEstimate(self,row,probs):
#	n = 2 case
# 		if row['Belief Val'] < cutoff:
# 			return row['Predicted State']
# 		else:
# 			return row['Alternate State']
		choice = self.determineIndex(probs, row['Belief Val'])
		if choice == 0:
			return row['Predicted State']
		else:
			return row['Alternate State_{0}'.format(choice-1)]

	def chooseRegionEstimate(self,row,probs):
#	n = 2 case
# 		if row['Belief Val'] < cutoff:
# 			return row['Predicted Region']
# 		else:
# 			return row['Alternate Region']
		choice = self.determineIndex(probs, row['Belief Val'])
		if choice == 0:
			return row['Predicted Region']
		else:
			return row['Alternate Region_{0}'.format(choice-1)]

	def makeGuess(self,df1,probs):
		#print('Inside the makeGuess function...')

		#print('Defining Lambdas...')
		state_fn = lambda x: self.chooseStateEstimate(x,probs)
		region_fn = lambda x: self.chooseRegionEstimate(x,probs)

		#print('Setting up new columns...')
		#df1['Alternate State'] = df2['Predicted State']
		#df1['Alternate Region'] = df2['Predicted Region']
		#print('Adding belief val...')
		df1['Belief Val'] = df1.apply(self.getRandom, axis=1)
		#print('Getting educated state...')
		df1['Educated State'] = df1.apply(state_fn, axis=1)
		#print('Getting educated region...')
		df1['Educated Region'] = df1.apply(region_fn, axis=1)
		#print('Returning dataframe...')
		#print('Exiting makeGuess')
		return df1

	def addMetricColumns(self,df):
		#print('Setting result stuff...')
		ed_states_results = (df['State']==df['Educated State']).apply(self.boolToInt)
		ed_regions_results = (df['Region']==df['Educated Region']).apply(self.boolToInt)
		orig_states_results = (df['State']==df['Predicted State']).apply(self.boolToInt)
		orig_regions_results = (df['Region']==df['Predicted Region']).apply(self.boolToInt)

		#print('calculations')
		tot_recs = len(ed_states_results)

		os_num = sum(orig_states_results)
		or_num = sum(orig_regions_results)
		os_pct = float(os_num)/float(tot_recs)
		or_pct = float(or_num)/float(tot_recs)

		es_num = sum(ed_states_results)
		er_num = sum(ed_regions_results)
		es_pct = float(es_num)/float(tot_recs)
		er_pct = float(er_num)/float(tot_recs)

		#print('Setting some columns to nums')
		df['States Guessed'] = os_num
		df['States Guessed %'] = os_pct
		df['Regions Guessed'] = or_num
		df['Regions Guessed %'] = or_pct
		df['Informed States Guessed'] = es_num
		df['Informed States Guessed %'] = es_pct
		df['Informed Regions Guessed'] = er_num
		df['Informed Regions Guessed %'] = er_pct

		#print('returning')
		return df

	def boolToInt(self,b):
		return int(b)


class TalkerAgent(BaseAgent):
	"""This agent will calculate classifications and send to another agent"""

	def __init__(self,receiverID,myID,myPass,mm,nt,nb,bs):
		self.receiver_id = spade.AID.aid(name=receiverID,
				addresses=["xmpp://"+receiverID])
		super(TalkerAgent,self).__init__(myID,myPass,mm,nt,nb,bs)

	class InformBehav(spade.Behaviour.OneShotBehaviour):

		def _process(self):
			#print("My receiver is called {0}".format(self.myAgent.receiver_id))
			receiver = self.myAgent.receiver_id

			self.msg = spade.ACLMessage.ACLMessage()
			self.msg.setPerformative("inform")
			self.msg.setOntology("classifierResults")
			self.msg.setLanguage("OWL-S")
			self.msg.addReceiver(receiver)

			testedSet = self.myAgent.classifier.perform_classification(TRAIN,TEST)
			#print('Tested!\nSetting content...')
			testedSet['Estimators'] = self.myAgent.classifier.num_estimators
			messageText = testedSet[['Predicted State','Predicted Region','Estimators']].to_json()
			#print('Final string is {0} chars long...'.format(len(messageText)))
			self.msg.setContent(messageText)
			#self.msg.setContent('ehllo guvnor')
			#print('Set!')
			#self.msg.setContent("test")

			#print(type(self.msg.getContent()))
			#print(self.msg.getContent()[:100])
			print('{0} sending message to {1}'.format(self.myAgent.getName(),receiver.getName()))
			self.myAgent.send(self.msg)
			print('Sent!')
			del(self.msg)
			del(testedSet)
			del(messageText)
			del(self.myAgent.classifier)
			#print('Killin\' it')
			agent_name = self.myAgent.getName()
			self.myAgent.stop()
			#print('Killed: {0}'.format(agent_name))

	def _setup(self):
		b = self.InformBehav()
		self.setDefaultBehaviour(b)

class ListenerAgent(BaseAgent):
	"""This agent will receive messages only"""
	def __init__(self,myID,myPass,numTalkers,mm,nt,nb,bs):
		self.numTalkers = numTalkers
		self.numHeard = 0
		self.probability = []
		self.dataFrame = None
		super(ListenerAgent,self).__init__(myID,myPass,mm,nt,nb,bs)
	
	class ReceiveBehav(spade.Behaviour.Behaviour):
		
		def _process(self):
			self.msg = None
			#print('Hello _process')

			self.msg = self._receive(True, MSG_WAIT)
			if self.msg:
				#print("Received!")


				myNewDataFrame = pd.read_json(self.msg.getContent())

				#print('Okay, chugging along')	
				if self.myAgent.numHeard == 0:
					self.myAgent.dataFrame = self.myAgent.classifier.perform_classification(TRAIN,TEST)
					#print('Muckling with probability array')
					self.myAgent.probability.append(self.myAgent.classifier.num_estimators)
					#print('Consider it muckled')
					
				myDataFrame = self.myAgent.dataFrame
					
				#print('Still up to no good...')
				self.myAgent.probability.append(myNewDataFrame['Estimators'].max())

				myDataFrame['Alternate State_{0}'.format(self.myAgent.numHeard)] = myNewDataFrame['Predicted State']
				myDataFrame['Alternate Region_{0}'.format(self.myAgent.numHeard)] = myNewDataFrame['Predicted Region']

				
				#print('numheard is {0}... incrementing...'.format(self.myAgent.numHeard))
				self.myAgent.numHeard += 1
				#print("numheard is now {0}, but numtalkers is {1}".format(self.myAgent.numHeard,self.myAgent.numTalkers))
				if self.myAgent.numHeard == self.myAgent.numTalkers:
					# decide on answers and spit out csv
					print(self.myAgent.probability)
					sprob = sum(self.myAgent.probability)
					alpha_prob = [float(i)/float(sprob) for i in self.myAgent.probability]
					print(alpha_prob)

					myDataFrame = self.myAgent.makeGuess(myDataFrame,alpha_prob)
					
					myDataFrame = self.myAgent.addMetricColumns(myDataFrame)
					
					with open('sim4out.csv','a') as simout:
						simout.write('{0},{1},{2},{3},{4},{5}\n'.format(
								myDataFrame['States Guessed'].max(),
								myDataFrame['Regions Guessed'].max(),
								myDataFrame['Informed States Guessed'].max(),
								myDataFrame['Informed Regions Guessed'].max(),
								myDataFrame['Informed States Guessed'].max()-myDataFrame['States Guessed'].max(),
								myDataFrame['Informed Regions Guessed'].max()-myDataFrame['Regions Guessed'].max(),
								
							))
					
					myDataFrame.to_csv('final.csv')
					os._exit(0)


				#print("Ontology: {0}".format(self.msg.getOntology()))
				
	def _setup(self):
		b = self.ReceiveBehav()

		classifier_template = spade.Behaviour.ACLTemplate()
		classifier_template.setOntology("classifierResults")
		msg_template = spade.Behaviour.MessageTemplate(classifier_template)

		self.addBehaviour(b,msg_template)

