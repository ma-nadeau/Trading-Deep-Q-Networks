# coding=utf-8

"""
Goal: Implementing a custom enhanced version of the Actor-Critic algorithm specialized
      for algorithmic trading.

Note: This code was expanded/modified based on the original code from the following authors:

Authors: Thibaut Théate and Damien Ernst (Modified for Actor-Critic)
Institution: University of Liège
"""

import math
import random
import copy
import datetime
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

from dates import startingDate, endingDate, splitingDate


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.softmax(self.model(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class TActorCritic:
    def __init__(self, observationSpace, actionSpace, hidden_dim=512, dropout=0.2, gamma=0.4, lr=0.00005):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(observationSpace, actionSpace, hidden_dim, dropout).to(self.device)
        self.critic = Critic(observationSpace, hidden_dim, dropout).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.actionSpace = actionSpace
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

    def chooseAction(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def learn(self, state, action_log_prob, reward, next_state, done, entropy):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([reward], dtype=torch.float, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.float, device=self.device)

        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        advantage = target - value

        actor_loss = -action_log_prob * advantage.detach()  - 0.001 * entropy
        critic_loss = F.smooth_l1_loss(value, target.detach())

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()



    def processReward(self, reward):
        return np.sign(reward) * np.log1p(abs(reward))

    def training(self, env, trainingParameters=[], verbose=True, rendering=False, plotTraining=True, showPerformance=True):
        episodes = trainingParameters[0] if trainingParameters else 50
        dataAugmentation = DataAugmentation()
        envList = dataAugmentation.generate(env)

        performanceTrain = []
        marketSymbol = env.marketSymbol
        startingDate = env.endingDate
        endingDate = '2020-1-1'
        money = env.data['Money'][0]
        stateLength = env.stateLength
        transactionCosts = env.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
        performanceTest = []
        score = np.zeros((len(envList), episodes))

        for episode in tqdm(range(episodes), disable=not verbose):
            for idx, env_i in enumerate(envList):
                coeffs = self.getNormalizationCoefficients(env_i)
                env_i.reset()
                startingPoint = random.randrange(len(env_i.data.index))
                env_i.setStartingPoint(startingPoint)
                state = self.processState(env_i.state, coeffs)
                done = False
                totalReward = 0

                while not done:
                    action, log_prob, entropy = self.chooseAction(state)
                    next_state_raw, reward, done, info = env_i.step(action)
                    next_state = self.processState(next_state_raw, coeffs)
                    reward = self.processReward(reward)
                    self.learn(state, log_prob, reward, next_state, done,entropy)
                    state = next_state
                    totalReward += reward

                score[idx][episode] = totalReward

            env_eval = self.testing(env, env)
            analyser = PerformanceEstimator(env_eval.data)
            sharpe_train = analyser.computeSharpeRatio()
            performanceTrain.append(sharpe_train)
            self.writer.add_scalar('Training performance (Sharpe Ratio)', sharpe_train, episode)

            env_test = self.testing(env, testingEnv)
            analyser = PerformanceEstimator(env_test.data)
            sharpe_test = analyser.computeSharpeRatio()
            performanceTest.append(sharpe_test)
            self.writer.add_scalar('Testing performance (Sharpe Ratio)', sharpe_test, episode)

            if episode % 10 == 0:  # print every 10 episodes
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    probs = self.actor(state_tensor).squeeze(0).cpu().numpy()
                print(f"[Episode {episode}] Action probs: {probs}")

        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(f'Figures_AC/{marketSymbol}_TrainingTestingPerformance.png')
            for i in range(len(envList)):
                self.plotTraining(score[i][:episodes], marketSymbol)

        if showPerformance:
            analyser = PerformanceEstimator(env.data)
            analyser.displayPerformance('TActorCritic',"".join(
                    [
                        "Figures_AC/",
                        str(marketSymbol),
                        f"_StartingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}_",
                        "_TrainingPerformance",
                    ]
                ),)

        if rendering:
            env.render('TAC')

        self.writer.close()
        return env

    def testing(self, trainEnv, testEnv, rendering=False, showPerformance=False):
        dataAugmentation = DataAugmentation()
        testFiltered = dataAugmentation.lowPassFilter(testEnv, 5)
        trainFiltered = dataAugmentation.lowPassFilter(trainEnv, 5)

        coeffs = self.getNormalizationCoefficients(trainFiltered)

        state = self.processState(testFiltered.reset(), coeffs)
        testEnv.reset()
        done = False

        while not done:
            action, _, _ = self.chooseAction(state)
            next_state, _, done, _ = testFiltered.step(action)
            testEnv.step(action)
            state = self.processState(next_state, coeffs)

        if rendering:
            testEnv.render('TAC')

        if showPerformance:
            analyser = PerformanceEstimator(testEnv.data)
            analyser.displayPerformance('TActorCritic',"".join(
                    [
                        "Figures_AC/",
                        str(testEnv.marketSymbol),
                        f"_StartingDate: {startingDate}_",
                        f"SplittingDate: {splitingDate}",
                        f"_EndingDate: {endingDate}_",
                        "_TestingPerformance",
                    ]
                ),)

        return testEnv

    def getNormalizationCoefficients(self, tradingEnv):
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()

        coefficients = []
        margin = 1
        returns = [abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]) for i in range(1, len(closePrices))]
        coefficients.append((0, np.max(returns) * margin))
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coefficients.append((0, np.max(deltaPrice) * margin))
        coefficients.append((0, 1))
        coefficients.append((np.min(volumes) / margin, np.max(volumes) * margin))
        return coefficients

    def processState(self, state, coefficients):
        closePrices = state[0]
        lowPrices = state[1]
        highPrices = state[2]
        volumes = state[3]

        returns = [(closePrices[i] - closePrices[i - 1]) / closePrices[i - 1] for i in range(1, len(closePrices))]
        state[0] = [((x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])) if coefficients[0][0] != coefficients[0][1] else 0 for x in returns]

        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(1, len(lowPrices))]
        state[1] = [((x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0])) if coefficients[1][0] != coefficients[1][1] else 0 for x in deltaPrice]

        closePricePosition = []
        for i in range(1, len(closePrices)):
            delta = abs(highPrices[i] - lowPrices[i])
            pos = abs(closePrices[i] - lowPrices[i]) / delta if delta != 0 else 0.5
            closePricePosition.append(pos)
        state[2] = [((x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0])) if coefficients[2][0] != coefficients[2][1] else 0.5 for x in closePricePosition]

        volumes = volumes[1:]
        state[3] = [((x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0])) if coefficients[3][0] != coefficients[3][1] else 0 for x in volumes]

        state = [item for sublist in state for item in sublist]
        return state

    def saveModel(self, actorPath, criticPath):
        torch.save(self.actor.state_dict(), actorPath)
        torch.save(self.critic.state_dict(), criticPath)

    def loadModel(self, actorPath, criticPath):
        self.actor.load_state_dict(torch.load(actorPath, map_location=self.device))
        self.critic.load_state_dict(torch.load(criticPath, map_location=self.device))

    def plotTraining(self, score, marketSymbol):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(f'Figures_AC/{marketSymbol}_TrainingResults.png')

    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        """
        GOAL: Plot the expected performance of the Actor-Critic trading agent.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number of episodes).
                - iterations: Number of training/testing iterations to compute the expected performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save initial weights
        initActor = copy.deepcopy(self.actor.state_dict())
        initCritic = copy.deepcopy(self.critic.state_dict())

        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = "2020-1-1"
        money = trainingEnv.data["Money"][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts

        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)

        for it in range(iterations):
            print(f"Expected performance iteration {it + 1}/{iterations}")
            for episode in tqdm(range(trainingParameters[0])):
                for env_i in trainingEnvList:
                    coeffs = self.getNormalizationCoefficients(env_i)
                    state = self.processState(env_i.reset(), coeffs)
                    done = False
                    while not done:
                        action, log_prob, entropy = self.chooseAction(state)
                        next_state_raw, reward, done, info = env_i.step(action)
                        next_state = self.processState(next_state_raw, coeffs)
                        reward = self.processReward(reward)
                        self.learn(state, log_prob, reward, next_state, done,entropy)
                        state = next_state

                env_eval = self.testing(trainingEnv, trainingEnv)
                analyser = PerformanceEstimator(env_eval.data)
                performanceTrain[episode][it] = analyser.computeSharpeRatio()
                self.writer.add_scalar("Training Sharpe", performanceTrain[episode][it], episode)

                env_test = self.testing(trainingEnv, testingEnv)
                analyser = PerformanceEstimator(env_test.data)
                performanceTest[episode][it] = analyser.computeSharpeRatio()
                self.writer.add_scalar("Testing Sharpe", performanceTest[episode][it], episode)

            if it < iterations - 1:
                trainingEnv.reset()
                testingEnv.reset()
                self.actor.load_state_dict(initActor)
                self.critic.load_state_dict(initCritic)
                self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.00005)
                self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.00005)

        expectedTrain = performanceTrain.mean(axis=1)
        expectedTest = performanceTest.mean(axis=1)
        stdTrain = performanceTrain.std(axis=1)
        stdTest = performanceTest.std(axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel="Expected Sharpe Ratio", xlabel="Episode")
        ax.plot(expectedTrain)
        ax.plot(expectedTest)
        ax.fill_between(range(len(expectedTrain)), expectedTrain - stdTrain, expectedTrain + stdTrain, alpha=0.25)
        ax.fill_between(range(len(expectedTest)), expectedTest - stdTest, expectedTest + stdTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        plt.savefig(f"Figures_AC/{marketSymbol}_TrainingTestingExpectedPerformance.png")
        self.writer.close()
        return trainingEnv
