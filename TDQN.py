# coding=utf-8

"""
Goal: Implementing a custom enhanced version of the DQN algorithm specialized
      to algorithmic trading.
      

Note: This code was expanded/modified based on the original code from the following authors:

Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège


"""

###############################################################################
################################### Imports ###################################
###############################################################################

import math
import random
import copy
import datetime

import numpy as np

from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt
from cpprb import PrioritizedReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

from dates import startingDate, endingDate, splitingDate



###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

# Default parameters related to the Experience Replay mechanism
capacity = 100000
batchSize = 32
experiencesRequired = 1000

# Default parameters related to the Deep Neural Network
numberOfNeurons = 512
dropout = 0.2

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000

# Default parameters regarding the sticky actions RL generalization technique
alpha = 0.1

# Default parameters related to preprocessing
filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1
rewardClipping = 1

# Default parameter related to the L2 Regularization
L2Factor = 0.000001

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


###############################################################################
############################### Class ReplayMemory ############################
###############################################################################


class ReplayMemory:
    """
    GOAL: Implementing the replay memory required for the Experience Replay
          mechanism of the DQN Reinforcement Learning algorithm.

    VARIABLES:  - memory: Data structure storing the experiences.

    METHODS:    - __init__: Initialization of the memory data structure.
                - push: Insert a new experience into the replay memory.
                - sample: Sample a batch of experiences from the replay memory.
                - __len__: Return the length of the replay memory.
                - reset: Reset the replay memory.
    """

    def __init__(self, capacity=capacity):
        """
        GOAL: Initializating the replay memory data structure.

        INPUTS: - capacity: Capacity of the data structure, specifying the
                            maximum number of experiences to be stored
                            simultaneously.

        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, nextState, done):
        """
        GOAL: Insert a new experience into the replay memory. An experience
              is composed of a state, an action, a reward, a next state and
              a termination signal.

        INPUTS: - state: RL state of the experience to be stored.
                - action: RL action of the experience to be stored.
                - reward: RL reward of the experience to be stored.
                - nextState: RL next state of the experience to be stored.
                - done: RL termination signal of the experience to be stored.

        OUTPUTS: /
        """

        self.memory.append((state, action, reward, nextState, done))

    def sample(self, batchSize):
        """
        GOAL: Sample a batch of experiences from the replay memory.

        INPUTS: - batchSize: Size of the batch to sample.

        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """

        state, action, reward, nextState, done = zip(
            *random.sample(self.memory, batchSize)
        )
        return state, action, reward, nextState, done

    def __len__(self):
        """
        GOAL: Return the capicity of the replay memory, which is the maximum number of
              experiences which can be simultaneously stored in the replay memory.

        INPUTS: /

        OUTPUTS: - length: Capacity of the replay memory.
        """

        return len(self.memory)

    def reset(self):
        """
        GOAL: Reset (empty) the replay memory.

        INPUTS: /

        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)


###############################################################################
################################### Class DQN #################################
###############################################################################
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=dropout):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            NoisyLinear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            NoisyLinear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


###############################################################################
################################ Class TDQN ###################################
###############################################################################


class TDQN:
    """
    GOAL: Implementing an intelligent trading agent based on the DQN
          Reinforcement Learning algorithm.

    VARIABLES:  - device: Hardware specification (CPU or GPU).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - capacity: Capacity of the experience replay memory.
                - batchSize: Size of the batch to sample from the replay memory.
                - targetNetworkUpdate: Frequency at which the target neural
                                       network is updated.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - policyNetwork: Deep Neural Network representing the RL policy.
                - targetNetwork: Deep Neural Network representing a target
                                 for the policy Deep Neural Network.
                - optimizer: Deep Neural Network optimizer (ADAM).
                - replayMemory: Experience replay memory.
                - epsilonValue: Value of the Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - iterations: Counter of the number of iterations.

    METHODS:    - __init__: Initialization of the RL trading agent, by setting up
                            many variables and parameters.
                - getNormalizationCoefficients: Retrieve the coefficients required
                                                for the normalization of input data.
                - processState: Process the RL state received.
                - processReward: Clipping of the RL reward received.
                - updateTargetNetwork: Update the target network, by transfering
                                       the policy network parameters.
                - chooseAction: Choose a valid action based on the current state
                                observed, according to the RL policy learned.
                - chooseActionEpsilonGreedy: Choose a valid action based on the
                                             current state observed, according to
                                             the RL policy learned, following the
                                             Epsilon Greedy exploration mechanism.
                - learn: Sample a batch of experiences and learn from that info.
                - training: Train the trading DQN agent by interacting with its
                            trading environment.
                - testing: Test the DQN agent trading policy on a new trading environment.
                - plotExpectedPerformance: Plot the expected performance of the intelligent
                                   DRL trading agent.
                - saveModel: Save the RL policy model.
                - loadModel: Load the RL policy model.
                - plotTraining: Plot the training results (score evolution, etc.).
                - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(
        self,
        observationSpace,
        actionSpace,
        numberOfNeurons=numberOfNeurons,
        dropout=dropout,
        gamma=gamma,
        learningRate=learningRate,
        targetNetworkUpdate=targetNetworkUpdate,
        epsilonStart=epsilonStart,
        epsilonEnd=epsilonEnd,
        epsilonDecay=epsilonDecay,
        capacity=capacity,
        batchSize=batchSize,
    ):
        """
        GOAL: Initializing the RL agent based on the DQN Reinforcement Learning
              algorithm, by setting up the DQN algorithm parameters as well as
              the DQN Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - targetNetworkUpdate: Update frequency of the target network.
                - epsilonStart: Initial (maximum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonEnd: Final (minimum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonDecay: Decay factor (exponential) of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - capacity: Capacity of the Experience Replay memory.
                - batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """

        # Initialise the random function with a new random seed
        random.seed(0)

        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device(
            "cuda:" + str(GPUNumber) if torch.cuda.is_available() else "cpu"
        )

        # Set the general parameters of the DQN algorithm
        self.gamma = gamma
        self.learningRate = learningRate
        self.targetNetworkUpdate = targetNetworkUpdate

        # Set the Experience Replay mechnism
        self.capacity = capacity
        self.batchSize = batchSize
        # self.replayMemory = ReplayMemory(capacity)
        self.replayMemory = PrioritizedReplayBuffer(
            size=capacity,
            env_dict={
                "state": {"shape": observationSpace},
                "next_state": {"shape": observationSpace},
                "action": {},
                "reward": {},
                "done": {},
            },
            alpha=0.5,  # Controls prioritization strength
        )

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.policyNetwork = DuelingDQN(
            observationSpace, actionSpace, numberOfNeurons
        ).to(self.device)
        self.targetNetwork = DuelingDQN(
            observationSpace, actionSpace, numberOfNeurons
        ).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(
            self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor
        )

        # Set the Epsilon-Greedy exploration technique
        self.epsilonValue = lambda iteration: epsilonEnd + (
            epsilonStart - epsilonEnd
        ) * math.exp(-1 * iteration / epsilonDecay)

        # Initialization of the iterations counter
        self.iterations = 0

        # Initialization of the tensorboard writer
        self.writer = SummaryWriter(
            "runs/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        )

        self.beta_start = 0.4
        self.beta_frames = 200000  # total steps to reach beta = 1.0

    def getNormalizationCoefficients(self, tradingEnv):
        """
        GOAL: Retrieve the coefficients required for the normalization
              of input data.

        INPUTS: - tradingEnv: RL trading environement to process.

        OUTPUTS: - coefficients: Normalization coefficients.
        """

        # Retrieve the available trading data
        tradingData = tradingEnv.data
        closePrices = tradingData["Close"].tolist()
        lowPrices = tradingData["Low"].tolist()
        highPrices = tradingData["High"].tolist()
        volumes = tradingData["Volume"].tolist()

        # Retrieve the coefficients required for the normalization
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [
            abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1])
            for i in range(1, len(closePrices))
        ]
        coeffs = (0, np.max(returns) * margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice) * margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes) / margin, np.max(volumes) * margin)
        coefficients.append(coeffs)

        return coefficients

    def processState(self, state, coefficients):
        """
        GOAL: Process the RL state returned by the environment
              (appropriate format and normalization).

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - state: Processed RL state.
        """

        # Normalization of the RL state
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        # 1. Close price => returns => MinMax normalization
        returns = [
            (closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]
            for i in range(1, len(closePrices))
        ]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [
                ((x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]))
                for x in returns
            ]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [
            abs(highPrices[i] - lowPrices[i]) for i in range(1, len(lowPrices))
        ]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [
                ((x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0]))
                for x in deltaPrice
            ]
        else:
            state[1] = [0 for x in deltaPrice]
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i] - lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i] - lowPrices[i]) / deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [
                ((x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0]))
                for x in closePricePosition
            ]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [
                ((x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0]))
                for x in volumes
            ]
        else:
            state[3] = [0 for x in volumes]

        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state

    def processReward(self, reward):
        """
        GOAL: Process the RL reward returned by the environment by clipping
              its value. Such technique has been shown to improve the stability
              the DQN algorithm.

        INPUTS: - reward: RL reward returned by the environment.

        OUTPUTS: - reward: Process RL reward.
        """

        return np.tanh(reward)  # or just return reward

    def updateTargetNetwork(self):
        """
        GOAL: Taking into account the update frequency (parameter), update the
              target Deep Neural Network by copying the policy Deep Neural Network
              parameters (weights, bias, etc.).

        INPUTS: /

        OUTPUTS: /
        """

        # Check if an update is required (update frequency)
        if self.iterations % targetNetworkUpdate == 0:
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def chooseAction(self, state):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            tensorState = torch.tensor(
                state, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            QValues = self.policyNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues

    def learning(self, batchSize=batchSize):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.

        INPUTS: batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """

        # Check that the replay memory is filled enough
        if self.replayMemory.get_stored_size() >= batchSize:
            # Set the Deep Neural Network in training mode
            self.policyNetwork.train()
            self.policyNetwork.reset_noise()
            self.targetNetwork.reset_noise()

            # Sample a batch of experiences from the replay memory
            # state, action, reward, nextState, done = self.replayMemory.sample(batchSize)
            # sample = self.replayMemory.sample(batchSize, beta=0.4)
            beta = min(
                1.0,
                self.beta_start
                + (1.0 - self.beta_start) * self.iterations / self.beta_frames,
            )
            sample = self.replayMemory.sample(batchSize, beta=beta)

            state = torch.tensor(
                sample["state"], dtype=torch.float32, device=self.device
            )
            nextState = torch.tensor(
                sample["next_state"], dtype=torch.float32, device=self.device
            )
            action = torch.tensor(
                sample["action"], dtype=torch.long, device=self.device
            )
            reward = torch.tensor(
                sample["reward"], dtype=torch.float32, device=self.device
            )
            done = torch.tensor(sample["done"], dtype=torch.float32, device=self.device)
            weights = torch.tensor(
                sample["weights"], dtype=torch.float32, device=self.device
            )
            indexes = sample["indexes"]

            # Initialization of Pytorch tensors for the RL experience elements
            # state = torch.tensor(state, dtype=torch.float, device=self.device)
            ##action = torch.tensor(action, dtype=torch.long, device=self.device)
            # reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            # nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
            # done = torch.tensor(done, dtype=torch.float, device=self.device)

            # Make sure action is a 1D tensor: shape [batch_size]
            if action.ndim > 1:
                action = action.squeeze()
            if action.ndim == 1:
                action = action.view(-1, 1)

            # Get Q-values from policy network: shape [batch_size, num_actions]
            q_values = self.policyNetwork(state)

            # Use gather to select Q-values for taken actions
            currentQValues = q_values.gather(1, action).squeeze(1)

            with torch.no_grad():
                # Get the best actions from the policy network (Double DQN)
                nextQ_policy = self.policyNetwork(nextState)
                nextActions = torch.argmax(
                    nextQ_policy, dim=1, keepdim=True
                )  # shape: [batch_size, 1]

                # Get Q-values from the target network
                nextQ_target = self.targetNetwork(nextState)
                nextQValues = nextQ_target.gather(1, nextActions).squeeze(
                    1
                )  # shape: [batch_size]

                reward = reward.view(-1)
                done = done.view(-1)
                nextQValues = nextQValues.view(-1)

                # Compute expected Q-values
                expectedQValues = reward + self.gamma * nextQValues * (1 - done)

            loss = (
                F.smooth_l1_loss(currentQValues, expectedQValues, reduction="none")
                * weights
            ).mean()

            # Compute the current Q values returned by the policy network
            # currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            # with torch.no_grad():
            #  nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
            #  nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
            #  expectedQValues = reward + gamma * nextQValues * (1 - done)

            # Compute the Huber loss
            # loss = F.smooth_l1_loss(currentQValues, expectedQValues)

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                td_errors = torch.abs(currentQValues - expectedQValues).cpu().numpy()
            self.replayMemory.update_priorities(indexes, td_errors)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.policyNetwork.parameters(), gradientClipping
            )

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            # If required, update the target deep neural network (update frequency)
            self.updateTargetNetwork()

            # Set back the Deep Neural Network in evaluation mode
            self.policyNetwork.eval()

    def training(
        self,
        trainingEnv,
        trainingParameters=[],
        verbose=False,
        rendering=False,
        plotTraining=False,
        showPerformance=False,
    ):
        """
        GOAL: Train the RL trading agent by interacting with its trading environment.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the training environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        """
        # Compute and plot the expected performance of the trading policy
        trainingEnv = self.plotExpectedPerformance(trainingEnv, trainingParameters, iterations=50)
        return trainingEnv
        """

        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = "2020-1-1"
            money = trainingEnv.data["Money"][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(
                marketSymbol,
                startingDate,
                endingDate,
                money,
                stateLength,
                transactionCosts,
            )
            performanceTest = []

        try:
            # If required, print the training progression
            if verbose:
                print(
                    "Training progression (hardware selected => "
                    + str(self.device)
                    + "):"
                )

            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0]), disable=not (verbose)):

                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):

                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0

                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0

                    # Interact with the training environment until termination
                    while done == 0:
                        self.policyNetwork.train()

                        # Choose an action according to the RL policy and the current RL state
                        # action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                        action, _, _ = self.chooseAction(state)
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)

                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        # self.replayMemory.push(state, action, reward, nextState, done)
                        self.replayMemory.add(
                            state=np.array(state, dtype=np.float32),
                            action=int(action),
                            reward=float(reward),
                            next_state=np.array(nextState, dtype=np.float32),
                            done=float(done),
                        )

                        # Trick for better exploration
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info["Reward"])
                        otherNextState = self.processState(info["State"], coefficients)
                        otherDone = info["Done"]
                        # self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                        self.replayMemory.add(
                            state=np.array(state, dtype=np.float32),
                            action=int(action),
                            reward=float(reward),
                            next_state=np.array(nextState, dtype=np.float32),
                            done=float(done),
                        )

                        # Execute the DQN learning procedure
                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0

                        # Update the RL state
                        state = nextState
                        previousAction = action

                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward

                    # Store the current training results
                    if plotTraining:
                        score[i][episode] = totalReward

                # Compute the current performance on both the training and testing sets
                if plotTraining:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar(
                        "Training performance (Sharpe Ratio)", performance, episode
                    )
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar(
                        "Testing performance (Sharpe Ratio)", performance, episode
                    )
                    testingEnv.reset()

        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv = self.testing(trainingEnv, trainingEnv)

        # If required, show the rendering of the trading environment
        if rendering:
            trainingEnv.render()

        # If required, plot the training results
        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(
                111, ylabel="Performance (Sharpe Ratio)", xlabel="Episode"
            )
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(
                "".join(
                    [
                        "Figures/",
                        str(marketSymbol),
                        f"_StatingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}_",
                        "_TrainingTestingPerformance",
                        ".png",
                    ]
                )
            )
            # plt.show()
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance(
                "TDQN",
                "".join(
                    [
                        "Figures/",
                        str(marketSymbol),
                        f"_StatingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}_",
                        "_TrainingPerformance",
                    ]
                ),
            )

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv

    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the RL agent trading policy on a new trading environment
              in order to assess the trading strategy performance.

        INPUTS: - trainingEnv: Training RL environment (known).
                - testingEnv: Unknown trading RL environment.
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:
            # Choose an action according to the RL policy and the current RL state
            self.policyNetwork.eval()

            action, _, QValues = self.chooseAction(state)

            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)

            # Update the new state
            state = self.processState(nextState, coefficients)

            # Storing of the Q values
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])

        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance(
                "TDQN",
                "".join(
                    [
                        "Figures/",
                        str(testingEnv.marketSymbol),
                        f"_StatingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}_",
                        "_TestingPerformance",
                    ]
                ),
            )

        return testingEnv

    def plotTraining(self, score, marketSymbol):
        """
        GOAL: Plot the training phase results
              (score, sum of rewards).

        INPUTS: - score: Array of total episode rewards.
                - marketSymbol: Stock market trading symbol.

        OUTPUTS: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel="Total reward collected", xlabel="Episode")
        ax1.plot(score)
        plt.savefig(
            "".join(
                [
                    "Figures/",
                    str(marketSymbol),
                    f"_StatingDate: {startingDate}_",
                    f"SplittingDate: {splitingDate}",
                    f"_EndingDate: {endingDate}_",
                    "TrainingResults",
                    ".png",
                ]
            )
        )
        # plt.show()

    def plotQValues(self, QValues0, QValues1, marketSymbol):
        """
        Plot sequentially the Q values related to both actions.

        :param: - QValues0: Array of Q values linked to action 0.
                - QValues1: Array of Q values linked to action 1.
                - marketSymbol: Stock market trading symbol.

        :return: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel="Q values", xlabel="Time")
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(["Short", "Long"])
        plt.savefig(
            "".join(
                [
                    "Figures/",
                    str(marketSymbol),
                    f"_StatingDate: {startingDate}_",
                    f"SplittingDate: {splitingDate}",
                    f"_EndingDate: {endingDate}" "_QValues",
                    ".png",
                ]
            )
        )
        # plt.show()

    def plotExpectedPerformance(
        self, trainingEnv, trainingParameters=[], iterations=10
    ):
        """
        GOAL: Plot the expected performance of the intelligent DRL trading agent.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - iterations: Number of training/testing iterations to compute
                              the expected performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Preprocessing of the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save the initial Deep Neural Network weights
        initialWeights = copy.deepcopy(self.policyNetwork.state_dict())

        # Initialization of some variables tracking both training and testing performances
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        # Initialization of the testing trading environment
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = "2020-1-1"
        money = trainingEnv.data["Money"][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(
            marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts
        )

        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))

        try:

            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):

                # Print the progression
                print(
                    "".join(
                        [
                            "Expected performance evaluation progression: ",
                            str(iteration + 1),
                            "/",
                            str(iterations),
                        ]
                    )
                )

                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(trainingParameters[0])):

                    # For each episode, train on the entire set of training environments
                    for i in range(len(trainingEnvList)):

                        # Set the initial RL variables
                        coefficients = self.getNormalizationCoefficients(
                            trainingEnvList[i]
                        )
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(
                            len(trainingEnvList[i].data.index)
                        )
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.processState(
                            trainingEnvList[i].state, coefficients
                        )
                        previousAction = 0
                        done = 0
                        stepsCounter = 0

                        # Interact with the training environment until termination
                        while done == 0:
                            self.policyNetwork.train()  # Ensure noise is active

                            # Choose an action according to the RL policy and the current RL state
                            action, _, _ = self.chooseAction(state)

                            # Interact with the environment with the chosen action
                            nextState, reward, done, info = trainingEnvList[i].step(
                                action
                            )

                            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            # self.replayMemory.push(state, action, reward, nextState, done)
                            self.replayMemory.add(
                                state=np.array(state, dtype=np.float32),
                                action=int(action),
                                reward=float(reward),
                                next_state=np.array(nextState, dtype=np.float32),
                                done=float(done),
                            )

                            # Trick for better exploration
                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info["Reward"])
                            otherDone = info["Done"]
                            otherNextState = self.processState(
                                info["State"], coefficients
                            )
                            # self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                            self.replayMemory.add(
                                state=np.array(state, dtype=np.float32),
                                action=int(action),
                                reward=float(reward),
                                next_state=np.array(nextState, dtype=np.float32),
                                done=float(done),
                            )

                            # Execute the DQN learning procedure
                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0

                            # Update the RL state
                            state = nextState
                            previousAction = action

                    # Compute both training and testing  current performances
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar(
                        "Training performance (Sharpe Ratio)",
                        performanceTrain[episode][iteration],
                        episode,
                    )
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar(
                        "Testing performance (Sharpe Ratio)",
                        performanceTest[episode][iteration],
                        episode,
                    )

                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations - 1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(
                        self.policyNetwork.parameters(),
                        lr=learningRate,
                        weight_decay=L2Factor,
                    )
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0

            iteration += 1

        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(
                np.mean(performanceTrain[episode][:iteration])
            )
            expectedPerformanceTest.append(
                np.mean(performanceTest[episode][:iteration])
            )
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)

        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(
                111, ylabel="Performance (Sharpe Ratio)", xlabel="Episode"
            )
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            plt.savefig(
                "".join(
                    [
                        "Figures/",
                        str(marketSymbol),
                        f"_StatingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}",
                        "_TrainingTestingPerformance",
                        str(i + 1),
                        ".png",
                    ]
                )
            )
            # plt.show()

        # Plot the expected performance of the intelligent DRL trading agent
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel="Performance (Sharpe Ratio)", xlabel="Episode")
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(
            range(len(expectedPerformanceTrain)),
            expectedPerformanceTrain - stdPerformanceTrain,
            expectedPerformanceTrain + stdPerformanceTrain,
            alpha=0.25,
        )
        ax.fill_between(
            range(len(expectedPerformanceTest)),
            expectedPerformanceTest - stdPerformanceTest,
            expectedPerformanceTest + stdPerformanceTest,
            alpha=0.25,
        )
        ax.legend(["Training", "Testing"])
        plt.savefig(
            "".join(
                [
                    "Figures/",
                    str(marketSymbol),
                    f"_StatingDate: {startingDate}_",
                    f"SplittingDate: {splitingDate}",
                    f"_EndingDate: {endingDate}",
                    "_TrainingTestingExpectedPerformance",
                    ".png",
                ]
            )
        )
        # plt.show()

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv

    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """

        torch.save(self.policyNetwork.state_dict(), fileName)

    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """

        self.policyNetwork.load_state_dict(
            torch.load(fileName, map_location=self.device)
        )
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def plotEpsilonAnnealing(self):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).

        INPUTS: /

        OUTPUTS: /
        """

        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(10 * epsilonDecay)])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        plt.savefig(
            "".join(
                [
                    "Figures/",
                    f"_StatingDate: {startingDate}_",
                    f"SplittingDate: {splitingDate}",
                    f"_EndingDate: {endingDate}_",
                    "EpsilonAnnealing",
                    ".png",
                ]
            )
        )
        # plt.show()
