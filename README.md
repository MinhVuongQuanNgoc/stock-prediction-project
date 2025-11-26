## COS30018 Intelligent Systems - Option B
This is a educational project for Swinburne University. Proceed with caution!


## Project Code are divide into tasks
- Task 1 - Runing given code 
- Task 2 - Download Data as CSV 
- Task 3 - Display data as candlesticks 
- Task 4 - Configurable DL models 
- Task 5 - Multistep and multivariate prediction 
- Task 6 - Ensembled models 
- Task 7 - Google Trend overlay 


## Setup and run ##
This is mainly a python project run in a virtual environment.
Download the code base to the local directory


**Virtual environment**
- Download Miniconda
- Create and active the virtual environment with python 3.9 using the Anaconda Promt
```
    conda create --name [env name] python=3.9
```
and
```
    conda activate [env name]
```

- Further description can be found [here](https://www.tensorflow.org/install/pip?lang=python3 "Miniconda Link")


## Instructions on how to run tasks ##
Open Anaconda promt and move towards the project directory and install dependencies using
```
  pip3 install -r requirements.txt
```

- Then open the file and choose the ticker, start date and end date, usually in the __main__ block. 
- Tickers list can be found at ticker_list.txt


## Task 2 ##
```
    cd Task-2
```
then
```
    python data_processing_1.py
```


## Task 3 ##
```
    cd Task-3
```
then
```
    python data_processing_2.py
```


## Task 4 ##
Change the hyperparameter configuration ID by changing TEST_ID in __main__

```
    cd Task-4
```
then
```
    python machine_learning_1.py
```


## Task 5 ##
- Note: Change the pred_steps argument in __main__ to change how many days ahead to predict
- change seq_len to how many days back are display since the predict point (2 small or 2 big caused the chart to be loose or condensed)
```
    cd Task-5
```
then
```
    python machine_learning_2.py
```
- Result is saved in output_dir (default named multi_output)


## Task 6 ##
```
    cd Task-6
```
- Note: Config the model used in model_set any combination of SARIMA, LSTM, GRU, BiLSTM, RF in __main__
- Example: model_set = ["SARIMA","RF","LSTM"]
```
    python machine_learning_3.py
```
- Output will be in the output_dir (default named output_ensemble) folder


## Task 7 ##
```
    cd Task-7
```
- Note: then to get the prediction data (config ensemble parameter similar to task 6)
```
    python machine_learning_3.py
```
- finally run this to get the trend overlay on top of ml3.py
NOTE!!! ALL OF THE FOLLOWING parameter need to be the same across all files
- ensemble_dir must have the same name with ml3.py output directory
- the keyword appropriately with the tick (example ticker NVDA should have the keyword "NVidia" or "NVidia Stock")
- pred_days to predict k days ahead
- Note: NodeAdjust ZOOM_DAYS to change how many days back are display since the predict point (2 small or 2 big caused the chart to be loose or condensed)
```
    python machine_learning_4.py
```


## Problems and notes
- Google Trends is inconsistent can return the 429 error to prevent spam. Run the file again until the request comes through

- There are problems with the visualization due to DataFrame contains leftover dtype=object values. YFinance occasionally returns columns like "Open" or "High" as mixed types on the first download (because cached CSV temporarily stores values as strings). This only occurs during the first run as in subsequent execution the cache is clean


=> TLDR: If first run fails, just run it again