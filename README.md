# ATP-Tennis-Match-Predictor
Machine Learning algorithm to predict results of ATP matches throughout the last two decades

# Steps
(1) Data taken from https://github.com/JeffSackmann/tennis_atp

(2) Filter data to get rid of unnecessary stats or stats that could skew the data (win/loss etc), these steps are done within merge.py to combine files and winner_to_binary.py to change the display of who won

(3) Collect overall player data (NOT USED IN ALGORITHM) to use in the future for better modeling, going along with the idea that better players will tend to win even if they are not predicted to by the algorithm

(4) Hot-one encoding to fill in missing values

(5) Analyze the effects of different variables within the data sets

(6) Run the random forest classifier algorithm, results: 63.55% accuracy on the test set

(7) Tune the algorithm, results: 65.03% accuracy on the test set

(8) Analyze early stopping

 
