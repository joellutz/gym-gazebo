#!/usr/bin/env python
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
import random
import numpy as np


class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists 
    that get returned as another list of dictionaries with each key corresponding to either 
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size) :
        indices = random.sample(np.arange(len(self.states)), min(size,len(self.states)) ) # sample some previously encountered states (128 indices)
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal) :
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        
        self.currentPosition += 1
# class Memory

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s'))

    """
    def __init__(self, input_shape, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_shape = input_shape
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
   
    def initNetworks(self, reloadModel=False):
        self.model = self.createModel(reloadModel)
    
    def saveModel(self):
        self.model.save_weights("model.h5", True)

    def createModel(self, reloadModel=False):
        # Recipe of deep reinforcement learning model
        model = Sequential()
        model.add(Convolution2D(16, kernel_size=3, input_shape=(self.input_shape[0], self.input_shape[1], 1), activation='relu'))
        model.add(Convolution2D(16, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.output_size))
        model.compile(optimizer=RMSprop(), loss='MSE')
        if(reloadModel):
            self.model.load_weights('model.h5')
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        # model.predict expects a 4-dim tensor (num_images, width, height, channels)
        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=3)
        predicted = self.model.predict(state) # array([[-66.83436, 1.3114333, ..., 17.54231]],      dtype=float32)
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size) # choose a random action (self.output_size = #actions)
        else :
            action = self.getMaxIndex(qValues) # choose the action with the highest q-value
        return action

    def selectActionByProbability(self, qValues, bias):
        """ not used here """
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        """ not used here """
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize):
        # Do not learn until we've got at least self.learnStart samples in our memory
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128: X = a state encountered, Y = resulting q-values for this state when applying the q-learn algorithm (reward + gamma * max(Q(s')))
            miniBatch = self.memory.getMiniBatch(miniBatchSize) # miniBatch[0] = {'action': 9081, 'isFinal': False, 'newState': array([[255, 255, 25...ype=uint8), 'reward': 0, 'state': array([[254, 255, 25...ype=uint8)}
            X_batch = np.empty((0,self.input_shape[0], self.input_shape[1], 1), dtype = np.uint8) # array([], shape=(0, 220, 300, 1), dtype=uint8)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)    # array([], shape=(0, 10000), dtype=float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state) # array([-15.969997  ,   9.60678   ,   3.7608333 , ...,  -9.308966  ,         6.0796537 ,  -0.03971964], dtype=float32)
                qValuesNewState = self.getQValues(newState) # array([-16.087975  ,   9.401747  ,   3.7499366 , ...,  -9.565148  ,         6.2042217 ,   0.03260362], dtype=float32)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal) # 43.634938087463375 = reward (0.0) + discountFactor (0.99) * max(qValuesNewState) (44.07)

                X_batch = np.append(X_batch, np.array([np.expand_dims(state.copy(), axis=2)]), axis=0) # array([[[254, 255, 254, ..., 255, 255, 255],        [251, 255, 255, ..., 255, 255, 236],        [255, 255, 255, ..., 255, 255, 255],        ...,        [251, 255, 255, ..., 255, 234, 255],        [255, 255, 255, ..., 255, 247, 255],        [255, 251, 255, ..., 255, 255, 244]]], dtype=uint8)
                Y_sample = qValues.copy()       # Y_sample = array([-0.16127397, -0.12661225], dtype=float32)
                Y_sample[action] = targetValue  # Y_sample = array([ 0.8819516 , -0.12661225], dtype=float32)
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            # train the network in order to match a state to correct (=experienced) q-values
            print("fitting model")
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)
        # for
    # learnOnMiniBatch
# class DeepQ