#import sys
#sys.stdout=open('C:/Users/andre/git/distai/output.txt','w')
#sys.stderr=open('C:/Users/andre/git/distai/errorlog.txt','w')
from pympler import muppy, summary
import argparse

parser = argparse.ArgumentParser(description = "Main function for RFC simulation")

parser.add_argument('-n',dest='num_agents',type=int,default=3,help='number of agents in simulation')
parser.add_argument('-m',dest='mix_method',type=str,default='gini',help='mix function {gini, entropy}')
parser.add_argument('-t',dest='num_trees',type=int,default=100,help='approx num of trees')
parser.add_argument('-b',dest='num_bins',type=int,default=8,help='number of deviations away from num_trees allowed')
parser.add_argument('-s',dest='bin_size',type=int,default=10,help='size of deviations')

args = parser.parse_args()


print("Beginning execution...") 
NUM_AGENTS = args.num_agents
MIX_METHOD = args.mix_method
NUM_TREES = args.num_trees
NUM_BINS = args.num_bins
BIN_SIZE = args.bin_size
print("Running {0}-agent simulation...".format(NUM_AGENTS))

import RFCAgents as rfca
print("imported...")

print("num agents set...")

agents = []
print("array initialized...")
agent_counter = 0
print("Counter set...")

agents.insert(0,rfca.ListenerAgent('listener@127.0.0.1','passw0rd',NUM_AGENTS-1,MIX_METHOD,NUM_TREES,NUM_BINS,BIN_SIZE))

while agent_counter < NUM_AGENTS-1:
	agents.insert(0,rfca.TalkerAgent('listener@127.0.0.1','talker{0}@127.0.0.1'.format(agent_counter),'passw0rd',MIX_METHOD,NUM_TREES,NUM_BINS,BIN_SIZE))
	agent_counter += 1




for a in agents[::-1]:
	print("Starting agent: {0}".format(a.getName()))
	with open('C:/Users/andre/git/distai/alives.txt','a') as appendfile:
		appendfile.write('Agents alive:{0}\n'.format([i.isRunning() for i in agents]))
	a.start()
    
