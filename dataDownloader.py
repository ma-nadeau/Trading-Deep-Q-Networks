# coding=utf-8

"""
Goal: Downloading financial data (related to stock markets) from diverse sources
      (Alpha Vantage, Yahoo Finance).
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import pandas as pd
import pandas_datareader as pdr
import requests

from io import StringIO

import yfinance as yf


###############################################################################
############################## Class AlphaVantage #############################
###############################################################################

class AlphaVantage:
    """
    GOAL: Downloading stock market data from the Alpha Vantage API. See the
          AlphaVantage documentation for more information.
    
    VARIABLES:  - link: Link to the Alpha Vantage website.
                - apikey: Key required to access the Alpha Vantage API.
                - datatype: 'csv' or 'json' data format.
                - outputsize: 'full' or 'compact' (only 100 time steps).
                - data: Pandas dataframe containing the stock market data.
                                
    METHODS:    - __init__: Object constructor initializing some variables.
                - getDailyData: Retrieve daily stock market data.
                - getIntradayData: Retrieve intraday stock market data.
                - processDataframe: Process the dataframe to homogenize the format.
    """

    def __init__(self):
        """
        GOAL: Object constructor initializing the class variables. 
        
        INPUTS: /      
        
        OUTPUTS: /
        """
        
        self.link = 'https://www.alphavantage.co/query'
        self.apikey = 'QII15G40KCLB7QQK'
        self.datatype = 'csv'
        self.outputsize = 'full'
        self.data = pd.DataFrame()
        
        
    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """
        GOAL: Downloading daily stock market data from the Alpha Vantage API. 
        
        INPUTS:     - marketSymbol: Stock market symbol.
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
          
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        
        # Send an HTTP request to the Alpha Vantage API
        payload = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': marketSymbol, 
                   'outputsize': self.outputsize, 'datatype': self.datatype, 
                   'apikey': self.apikey}
        response = requests.get(self.link, params=payload)
        
        # Process the CSV file retrieved
        csvText = StringIO(response.text)
        print(csvText)
        data = pd.read_csv(csvText, index_col='timestamp')
        
        # Process the dataframe to homogenize the output format
        self.data = self.processDataframe(data)
        if(startingDate != 0 and endingDate != 0):
            self.data = self.data.loc[startingDate:endingDate]

        return self.data
        
        
    def getIntradayData(self, marketSymbol, startingDate, endingDate, timePeriod=60):
        """
        GOAL: Downloading intraday stock market data from the Alpha Vantage API. 
        
        INPUTS:     - marketSymbol: Stock market symbol. 
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
                    - timePeriod: Time step of the stock market data (in seconds).
          
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        
        # Round the timePeriod value to the closest accepted value
        possiblePeriods = [1, 5, 15, 30, 60]
        timePeriod = min(possiblePeriods, key=lambda x:abs(x-timePeriod))
        
        # Send a HTTP request to the AlphaVantage API
        payload = {'function': 'TIME_SERIES_INTRADAY', 'symbol': marketSymbol, 
                   'outputsize': self.outputsize, 'datatype': self.datatype, 
                   'apikey': self.apikey, 'interval': str(timePeriod)+'min'}
        response = requests.get(self.link, params=payload)
        
        # Process the CSV file retrieved
        csvText = StringIO(response.text)
        data = pd.read_csv(csvText, index_col='timestamp')
        
        # Process the dataframe to homogenize the output format
        self.data = self.processDataframe(data)
        if(startingDate != 0 and endingDate != 0):
            self.data = self.data.loc[startingDate:endingDate]

        return self.data
    
    
    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # Reverse the order of the dataframe (chronological order)
        dataframe = dataframe[::-1]

        # Remove useless columns
        dataframe['close'] = dataframe['adjusted_close']
        del dataframe['adjusted_close']
        del dataframe['dividend_amount']
        del dataframe['split_coefficient']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe.rename(index=str, columns={"open": "Open",
                                                         "high": "High", 
                                                         "low": "Low",
                                                         "close": "Close",
                                                         "volume": "Volume"})
        # Adjust the format of the index values
        dataframe.index = dataframe.index.map(pd.Timestamp)

        return dataframe



###############################################################################
########################### Class YahooFinance ################################
###############################################################################

class YahooFinance:   
    """
    GOAL: Downloading stock market data from the Yahoo Finance API. See the
          pandas.datareader documentation for more information.
    
    VARIABLES:  - data: Pandas dataframe containing the stock market data.
                                
    METHODS:    - __init__: Object constructor initializing some variables.
                - getDailyData: Retrieve daily stock market data.
                - processDataframe: Process a dataframe to homogenize the
                                    output format.
    """
    

    def __init__(self):
        """
        GOAL: Object constructor initializing the class variables. 
        
        INPUTS: /      
        
        OUTPUTS: /
        """
        
        self.data = pd.DataFrame()

    
    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """
        GOAL: Downloding daily stock market data from the Yahoo Finance API. 
        
        INPUTS:     - marketSymbol: Stock market symbol.
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
          
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        
        #data = pdr.data.DataReader(marketSymbol, 'yahoo', startingDate, endingDate)
        
        data = yf.download(marketSymbol, start=startingDate, end=endingDate, auto_adjust=False)
        
        self.data = self.processDataframe(data)
        print('Data processed:', self.data)
        return self.data

   

    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # # Remove useless columns
        # dataframe['Close'] = dataframe['Adj Close']
        # del dataframe['Adj Close']
        
        # # Adapt the dataframe index and column names
        # dataframe.index.names = ['Timestamp']
        # dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        # return dataframe
          # Reset index to move timestamp into a column
        dataframe = dataframe.reset_index()

        # Flatten column levels if multi-indexed
        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe.columns = dataframe.columns.get_level_values(0)

        # Replace 'Close' with 'Adj Close' if it exists
        if 'Adj Close' in dataframe.columns:
            dataframe['Close'] = dataframe['Adj Close']
            dataframe.drop(columns=['Adj Close'], inplace=True)

        # Ensure column order and names
        dataframe = dataframe[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        dataframe.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

        dataframe.set_index('Timestamp', inplace=True)
        
        return dataframe

    
###############################################################################
############################# Class CSVHandler ################################
###############################################################################
    
class CSVHandler:
    """
    GOAL: Converting "Pandas dataframe" <-> "CSV file" (bidirectional).
    
    VARIABLES: /
                                
    METHODS:    - dataframeToCSV: Saving a dataframe into a CSV file.
                - CSVToDataframe: Loading a CSV file into a dataframe.
    """
    
    
    def dataframeToCSV(self, name, dataframe):
        """
        GOAL: Saving a dataframe into a CSV file.
        
        INPUTS:     - name: Name of the CSV file.   
                    - dataframe: Pandas dataframe to be saved.
          
        OUTPUTS: /
        """
        
        path = name + '.csv'
        dataframe.to_csv(path)
        
        
    def CSVToDataframe(self, name):
        """
        GOAL: Loading a CSV file into a dataframe.
        
        INPUTS:     - name: Name of the CSV file.   
          
        OUTPUTS:    - dataframe: Pandas dataframe loaded.
        """
        
        path = name + '.csv'
        return pd.read_csv(path,
                           header=0,
                           index_col='Timestamp',
                           parse_dates=True)
        
    
    def replace_csv_header(self, filepath, new_header='Timestamp,Open,High,Low,Close,Volume', skip_lines=3, output_path=None):
        """
        Replaces the first `skip_lines` of a CSV file with `new_header`.
        
        Args:
            filepath (str): Path to the original CSV file.
            new_header (str): The new header line (without newline at end).
            skip_lines (int): Number of lines to skip at the beginning.
            output_path (str, optional): Path to save the modified file. 
                                        If None, overwrites the original file.
        """
        output_path = output_path or filepath
    
        with open(filepath, 'r') as f:

            lines = f.readlines()
            if lines[0].strip() == new_header.strip():
                return  # Do nothing if the first line is already the new header
            lines = lines[skip_lines:]

        with open(output_path, 'w') as f:
            f.write(new_header.strip() + '\n')
            f.writelines(lines)
        