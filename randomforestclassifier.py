from pandas.errors import SettingWithCopyWarning
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine
import optuna
warnings.filterwarnings('ignore')

#get csv files into dataframes
training_data = pd.read_csv ('2015_2021_training_r.csv')
testing_data = pd.read_csv ('2022_testing_r.csv')
validation_data = pd.read_csv ('2021_validation_r.csv')

#isolate the target variable
training_label = np.array(training_data['winner_name'])
testing_label = np.array(testing_data['winner_name'])
validation_label = np.array(validation_data['winner_name'])

#get necessary variables for training and testing data
features_train = training_data[['tourney_name','surface','draw_size','tourney_level','tourney_date','match_num','p1_id','p1_seed','p1_name','p1_hand','p1_ht','p1_age','p2_id','p2_seed','p2_name','p2_hand','p2_ht','p2_age','p1_rank','p1_rank_points','p2_rank','p2_rank_points']]
features_val = validation_data[['tourney_name','surface','draw_size','tourney_level','tourney_date','match_num','p1_id','p1_seed','p1_name','p1_hand','p1_ht','p1_age','p2_id','p2_seed','p2_name','p2_hand','p2_ht','p2_age','p1_rank','p1_rank_points','p2_rank','p2_rank_points']]
features_test = testing_data[['tourney_name','surface','draw_size','tourney_level','tourney_date','match_num','p1_id','p1_seed','p1_name','p1_hand','p1_ht','p1_age','p2_id','p2_seed','p2_name','p2_hand','p2_ht','p2_age','p1_rank','p1_rank_points','p2_rank','p2_rank_points']]


#hotencode categorical variables
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
features_train["tourney_name"] = ord_enc.fit_transform(features_train[["tourney_name"]])
features_train["surface"] = ord_enc.fit_transform(features_train[["surface"]])
features_train["tourney_date"] = ord_enc.fit_transform(features_train[["tourney_date"]])
features_train["tourney_level"] = ord_enc.fit_transform(features_train[["tourney_level"]])
features_train["p1_name"] = ord_enc.fit_transform(features_train[["p1_name"]])
features_train["p2_name"] = ord_enc.fit_transform(features_train[["p2_name"]])
features_train["p1_hand"] = ord_enc.fit_transform(features_train[["p1_hand"]])
features_train["p2_hand"] = ord_enc.fit_transform(features_train[["p2_hand"]])

features_val["tourney_name"] = ord_enc.fit_transform(features_val[["tourney_name"]])
features_val["surface"] = ord_enc.fit_transform(features_val[["surface"]])
features_val["tourney_date"] = ord_enc.fit_transform(features_val[["tourney_date"]])
features_val["tourney_level"] = ord_enc.fit_transform(features_val[["tourney_level"]])
features_val["p1_name"] = ord_enc.fit_transform(features_val[["p1_name"]])
features_val["p2_name"] = ord_enc.fit_transform(features_val[["p2_name"]])
features_val["p1_hand"] = ord_enc.fit_transform(features_val[["p1_hand"]])
features_val["p2_hand"] = ord_enc.fit_transform(features_val[["p2_hand"]])

features_test["tourney_name"] = ord_enc.fit_transform(features_test[["tourney_name"]])
features_test["surface"] = ord_enc.fit_transform(features_test[["surface"]])
features_test["tourney_date"] = ord_enc.fit_transform(features_test[["tourney_date"]])
features_test["tourney_level"] = ord_enc.fit_transform(features_test[["tourney_level"]])
features_test["p1_name"] = ord_enc.fit_transform(features_test[["p1_name"]])
features_test["p2_name"] = ord_enc.fit_transform(features_test[["p2_name"]])
features_test["p1_hand"] = ord_enc.fit_transform(features_test[["p1_hand"]])
features_test["p2_hand"] = ord_enc.fit_transform(features_test[["p2_hand"]])

#impute missing v alues
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# Num_vars is the list of numerical variables
X_train = pd.DataFrame(features_train)
features_train = imputer.fit_transform(features_train)
features_val = imputer.fit_transform(features_val)
features_test = imputer.fit_transform(features_test)

#traing and test
clf = RandomForestClassifier()
clf.fit(features_train, training_label)
preds = clf.predict(features_test)
print("Training accuracy: ", clf.score(features_train, training_label))
print("Testing accuracy: ", clf.score(features_test, testing_label))

print("Model Features Ranked: ")
print(pd.DataFrame(clf.feature_importances_, index=X_train.columns.values).sort_values(by=0, ascending=False))


clf_4 = RandomForestClassifier(max_depth=4)
clf_4.fit(features_train, training_label)
print("Accuracy on training set only using top 4 features: %f" % clf_4.score(features_train, training_label))
print("Accuracy on test set only using top 4 features: %f" % clf_4.score(features_test, testing_label))



def objective(trial):

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }

    # Fit the model
    optuna_model = XGBClassifier(**params)
    optuna_model.fit(features_train, training_label)

    # Make predictions
    y_pred = optuna_model.predict(features_test)

    # Evaluate predictions
    accuracy = accuracy_score(testing_label, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))
print('  Params: ')

for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

params = trial.params
model = XGBClassifier(**params)
model.fit(features_train, training_label)
y_pred = model.predict(features_test)
accuracy = accuracy_score(testing_label, y_pred)
print("Accuracy on test data after tuning: %.2f%%" % (accuracy * 100.0))
print(classification_report(testing_label, y_pred))



#now use early stopping on best hypertuned parameters
cls = XGBClassifier(**params)
cls.fit(features_train, training_label, eval_set=[(features_train, training_label), (features_val, validation_label)],
            early_stopping_rounds=20)

y_pred = cls.predict(features_test)
accuracy = accuracy_score(testing_label, y_pred)
print("Accuracy on test data after early stopping and tuning: %.2f%%" % (accuracy * 100.0))
print(classification_report(testing_label, y_pred))

results = cls.evals_result()

print("The best tree length: ", model.best_ntree_limit)


plt.figure(figsize=(10,7))
plt.plot(results["validation_0"]["logloss"], label="Training loss")
plt.plot(results["validation_1"]["logloss"], label="Validation loss")
plt.axvline(21, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.legend()
plt.savefig("NumberOfTrees.svg")