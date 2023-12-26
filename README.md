# ATP-Tennis-Match-Predictor
Machine Learning algorithm to predict results of ATP matches throughout the last two decades

# Results
Predicted winner of match correctly ~65% of the time

<img width="989" alt="Screenshot 2023-12-26 at 6 20 04 PM" src="https://github.com/alexcole74/ATP-Tennis-Match-Predictor/assets/154842337/81c53c11-2efb-4caa-99b0-5c6faf54cb21">

# Setup
Follow the steps below that are in the jupyter notebook file, will need to change directory files to ensure data is being accessed correctly

# Steps
(1) Data taken from https://github.com/JeffSackmann/tennis_atp

(2) Filter data to get rid of unnecessary stats or stats that could skew the data (win/loss etc), these steps are done within merge.py to combine files and winner_to_binary.py to change the display of who won

(3) Collect overall player data (NOT USED IN ALGORITHM) to use in the future for better modeling, going along with the idea that better players will tend to win even if they are not predicted to by the algorithm

(4) Hot-one encoding to fill in missing values

(5) Analyze the effects of different variables within the data sets

<img width="379" alt="Screenshot 2023-12-26 at 6 20 51 PM" src="https://github.com/alexcole74/ATP-Tennis-Match-Predictor/assets/154842337/408f0bb6-f140-4a35-87e4-6ce608b996f1">

(6) Run the random forest classifier algorithm, results: 63.55% accuracy on the test set

(7) Tune the algorithm, results: 65.03% accuracy on the test set

(8) Analyze early stopping

 <img width="698" alt="Screenshot 2023-12-26 at 6 21 27 PM" src="https://github.com/alexcole74/ATP-Tennis-Match-Predictor/assets/154842337/eb6a2c57-b60d-4cc4-8e7d-f85c0827748c">

