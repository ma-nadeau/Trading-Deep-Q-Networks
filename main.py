# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='S&P 500', type=str, help="Name of the stock (market)")
    print(parser)
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategyAi = args.strategy
    stock = args.stock

    # Training and testing of the trading strategy specified for the stock (market) specified
    simulator.simulateNewStrategy('TDQN', stock, numberOfEpisodes=50, showPerformance=True, saveStrategy=False)
    
    # simulator.displayTestbench()
    # simulator.analyseTimeSeries(stock)
    # simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    # simulator.simulateExistingStrategy('TDQN', stock)
    # simulator.evaluateStrategy('TDQN', saveStrategy=False)
    # simulator.evaluateStock('Apple')
    
