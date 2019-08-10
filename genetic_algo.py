import numpy as np
import random
import torch
from game_ga import train

'''
list of hyperparams:
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 200000
TARGET_UPDATE = 10000
LEARNING_FREQ = 1
LEARNING_START = 5000
MEMORY_SIZE = 10000
OPTIM_LR = 0.0025
'''
class DeepQNetworks():
    def __init__(self):
        self.epochs = np.random.randint(1000, 5000)
        self.batch_size = np.random.randint(32, 64)
        self.gamma = random.uniform(0.95, 0.99)
        self.eps_start = 1
        self.eps_end = random.uniform(0.01, 0.1)
        self.target_update = np.random.randint(10,1000)
        self.learning_freq = 1
        self.memory_size = np.random.randint(500, 1000)
        self.learning_start = np.random.randint(0, 0.5*self.memory_size)
        self.optim_lr = random.uniform(0.0025, 0.025)
        self.eval = 0 #last score
        self.penalty1 = np.random.randint(10, 100)
        self.penalty2 = np.random.randint(30, 500)
        self.penalty3 = np.random.randint(1000, 10000)

    def hyperparams(self):
        hyperparams = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'target_update': self.target_update,
            'learning_freq': self.learning_freq,
            'learning_start': self.learning_start,
            'memory_size': self.memory_size,
            'optim_lr': self.optim_lr,
            'eval': self.eval,
            'penalty1': self.penalty1,
            'penalty2':self.penalty2,
            'penalty3':self.penalty3
        }
        return hyperparams

#create a set of initial parents
def population(num):
    return [DeepQNetworks() for _ in range(num)]

def calc_pop_fitness(dqn_networks):
    counter = 1
    for network in dqn_networks:
        hyperparams = network.hyperparams()
        epochs = hyperparams['epochs']
        batch_size = hyperparams['batch_size']
        gamma = hyperparams['gamma']
        eps_start = hyperparams['eps_start']
        eps_end = hyperparams['eps_end']
        target_update = hyperparams['target_update']
        learning_freq = hyperparams['learning_freq']
        learning_start = hyperparams['learning_start']
        memory_size = hyperparams['memory_size']
        optim_lr = hyperparams['optim_lr']

        dqn_score, state_dict = train(hyperparams)
        network.eval = dqn_score
        print("Latest model score: ")
        print(network.eval)

        torch.save(state_dict, r"C:\Users\Lector\Desktop\breakout\weights\policy" + str(counter) + ".pt")
        counter += 1

    return dqn_networks

#select the top 20% of the dqn models
def selection(dqn_networks):
    dqn_networks = sorted(dqn_networks, key=lambda dqn_model: dqn_model.eval, reverse=True)
    dqn_networks = dqn_networks[:int(0.2 * len(dqn_networks))]

    return dqn_networks

'''
self.epochs = np.random.randint(1000, 5000)
self.batch_size = np.random.randint(32, 64)
self.gamma = random.uniform(0.95, 0.99)
self.eps_start = 1
self.eps_end = random.uniform(0.01, 0.1)
self.target_update = 10000
self.learning_freq = 1
self.learning_start = np.random.randint(500, 5000)
self.memory_size = np.random.randint(1000, 5000)
self.optim_lr = random.uniform(0.0025, 0.025)
self.eval = 0 #last score
self.penalty1 = np.random.randint(10, 20)
self.penalty2 = np.random.randint(30, 50)
self.penalty3 = np.random.randint(1000, 10000)
'''
def crossover(dqn_networks, pop_num):
    children = []
    for _ in range(int((pop_num - len(dqn_networks)) / 2)):
        parent1 = random.choice(dqn_networks)
        parent2 = random.choice(dqn_networks)
        child1 = DeepQNetworks()
        child2 = DeepQNetworks()
        # Crossing over parent hyper-params
        #number of eps_decay
        child1.epochs = int(parent1.epochs*0.4) + int(parent2.epochs*0.4) + int(child1.epochs*0.2)
        child2.epochs = int(parent1.epochs*0.3) + int(parent2.epochs*0.3) + int(child2.epochs*0.4)

        child1.batch_size = int(parent1.batch_size*0.4) + int(parent2.batch_size*0.4) + int(child1.batch_size*0.2)
        child2.batch_size = int(parent1.batch_size*0.3) + int(parent2.batch_size*0.3) + int(child1.batch_size*0.4)

        child1.gamma = int(parent1.gamma*0.4) + int(parent2.gamma*0.4) + int(child1.gamma*0.2)
        child2.gamma = int(parent1.gamma*0.3) + int(parent2.gamma*0.3) + int(child1.gamma*0.4)

        child1.eps_end = int(parent1.eps_end*0.4) + int(parent2.eps_end*0.4) + int(child1.eps_end*0.2)
        child2.eps_end = int(parent1.eps_enda*0.3) + int(parent2.eps_end*0.3) + int(child1.eps_end*0.4)

        child1.memory_size = int(parent1.memory_size*0.4) + int(parent2.memory_size*0.4) + int(child1.memory_size*0.2)
        child2.memory_size = int(parent1.memory_size*0.3) + int(parent2.memory_size*0.3) + int(child1.memory_size*0.4)

        child1.learning_start = int(parent1.learning_start*0.4) + int(parent2.learning_start*0.4) + int(child1.learning_start*0.2)
        child2.learning_start = int(parent1.learning_start*0.3) + int(parent2.learning_start*0.3) + int(child1.learning_start*0.4)

        child1.optim_lr = int(parent1.optim_lr*0.4) + int(parent2.optim_lr*0.4) + int(child1.optim_lr*0.2)
        child2.optim_lr = int(parent1.optim_lr*0.3) + int(parent2.optim_lr*0.3) + int(child1.optim_lr*0.4)

        child1.target_update = int(parent1.target_update*0.4) + int(parent2.target_update*0.4) + int(child1.target_update*0.2)
        child2.target_update = int(parent1.target_update*0.3) + int(parent2.target_update*0.3) + int(child1.target_update*0.4)

        child1.penalty1 = int(parent1.penalty1*0.4) + int(parent2.penalty1*0.4) + int(child1.penalty1*0.2)
        child2.penalty1 = int(parent1.penalty1*0.3) + int(parent2.penalty1*0.3) + int(child1.penalty1*0.4)

        child1.penalty2 = int(parent1.penalty2*0.4) + int(parent2.penalty2*0.4) + int(child1.penalty2*0.2)
        child2.penalty2 = int(parent1.penalty2*0.3) + int(parent2.penalty2*0.3) + int(child1.penalty2*0.4)

        child1.penalty3 = int(parent1.penalty3*0.4) + int(parent2.penalty3*0.4) + int(child1.penalty3*0.2)
        child2.penalty3 = int(parent1.penalty3*0.3) + int(parent2.penalty3*0.3) + int(child1.penalty3*0.4)

        children.append(child1)
        children.append(child2)

    dqn_networks.extend(children)

    return dqn_networks

def mutate(dqn_networks):
    for network in dqn_networks:
        if np.random.uniform(0, 1) <= 0.1:
            network.epochs += np.random.randint(0,100)
            network.learning_start += np.random.randint(0,10)
            network.optim_lr += random.uniform(0,0.001)

    return dqn_networks

def main():
    #assuming that the initial population is 25
    pop_num = 20
    dqn_networks = population(pop_num)

    #assuming that we train dqn for 5 generations of populations
    gen_num = 100
    for gen in range(gen_num):
        print ('Generation {}'.format(gen+1))

        dqn_network = calc_pop_fitness(dqn_networks)
        dqn_network = selection(dqn_networks)
        dqn_network = crossover(dqn_networks, pop_num)
        dqn_network = mutate(dqn_networks)

        counter = 1
        for network in dqn_networks:
            if network.eval > 250: #the AI successfully plays the game
                print ('Threshold met')
                print (network.hyperparams())
                print ('Best accuracy: {}'.format(network.eval))
                exit(0)

            print('Network' + str(counter) + ':')
            print(network.hyperparams())
            counter += 1

if __name__ == '__main__':
    main()
