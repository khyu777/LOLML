import pandas as pd
import os
from sklearn import linear_model, preprocessing, svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import xgboost as xgb
import warnings
import random
warnings.filterwarnings("ignore", category=DeprecationWarning)

#set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('Data/Prediction'):
    os.mkdir('Data/Prediction')

#run a single match
def match(data, team1, team2, model, random_scale=1):
	match = pd.DataFrame(columns = ['sp1', 'gp1', 'p1', 'sp2', 'gp2', 'p2'], index=[0])

	gd1 = data[data.Team == team1]['GD'].iloc[0]
	kd1 = data[data.Team == team1]['KD'].iloc[0]
	td1 = data[data.Team == team1]['TD'].iloc[0]
	dd1 = data[data.Team == team1]['DD'].iloc[0]
	bd1 = data[data.Team == team1]['BD'].iloc[0]
	rhd1 = data[data.Team == team1]['RHD'].iloc[0]

	match['sp1'] = data[data.Team == team1]['SPF'].iloc[0]
	match['sp2'] = data[data.Team == team2]['SPF'].iloc[0]
	match['sp2'] = -match['sp2']
	if len(data.columns) == 13:
		match['gp1'] = data[data.Team == team1]['GPF'].iloc[0]
		match['gp2'] = data[data.Team == team2]['GPF'].iloc[0]
		match['gp2'] = -match['gp2']
	elif len(data.columns) == 14:
		match['p1'] = data[data.Team == team1]['P'].iloc[0]
		match['p2'] = data[data.Team == team2]['P'].iloc[0]
		match['p2'] = -match['p2']
	
	match['gd1'] = np.random.normal(gd1, 3)
	match['kd1'] = np.random.normal(kd1, 3)
	match['td1'] = np.random.normal(td1, 3)
	match['dd1'] = np.random.normal(dd1, 3)
	match['bd1'] = np.random.normal(bd1, 3)
	match['rhd1'] = np.random.normal(rhd1, 3)
	match = match.dropna(axis=1)
	
	match_array = match.values
	
	prediction = model.predict(match_array)
	
	winner = None
	
	if prediction == 1:
		winner = team1
	elif prediction == 0:
		winner = team2
	return winner

#match simulation
def simulate_matches(team, model, n_matches=1000):
	winner = team[0]
	print('-------------------------')
	print()
	winners = []
	place = []
	for i in range(1,5):
		match_results = []
		for j in range(n_matches):
			match_results.append(match(standings, winner, team[i], model))
			match_results.append(match(standings, team[i], winner, model))

		team1_proba = match_results.count(winner)/len(match_results)*100
		team2_proba = match_results.count(team[i])/len(match_results)*100

		print(winner, ': ', team1_proba)
		print(team[i], ': ', team2_proba)

		if team1_proba > team2_proba:
			winner = winner
			loser = team[i]
		else:
			loser = winner
			winner = team[i]
		winners.append(winner)
		if i < 5:
			place.append(loser)
		'''
		p_list = []
		for num in range(len(match_results)):
			print(num)
			a = match_results[:num].count(winner) / (num+1) * 100
			b = match_results[:num].count(team[i]) / (num+1) * 100
			p_list.append(a - b)
		
		plt.plot(p_list)
		plt.show()
		'''

		print('Winner: ', winner)
		print()
		print('-------------------------')
		print()
	place.append(winners[-1])
	print(place)
	return winners, place

#read files and store names
path = 'Data/'
files = []
for file in os.listdir(path):
	file_name, file_ext = os.path.splitext(file)
	if file_ext == '.csv' and 'Playoffs' not in file_name and 'Standings' not in file_name:
		files.append(file_name)
league = files[0][0:3]

results = []
lacc = []
racc = []
sacc = []
xacc = []
places = []
for file in files:
	print(file)
	#read data
	regular = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + '.csv', index_col=0)
	#playoff = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Playoffs.csv', index_col=0)
	standings = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Standings.csv', index_col=0)
	standings = standings.iloc[0:5]

	teams = standings['Team'].iloc[::-1]
	teams = teams.reset_index(drop=True)

	#xgboost testing
	X = regular.iloc[:, 3:-1]
	y = regular.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#normalize data
	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(X_train)
	X_train = pd.DataFrame(np_scaled).values
	np_test_scaled = min_max_scaler.fit_transform(X_test)
	X_test = pd.DataFrame(np_test_scaled).values

	#train XGB model
	xc = xgb.XGBClassifier()
	xc.fit(X_train, y_train)
	y_pred = xc.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	xacc.append(accuracy)

	#train logistic regression model
	lr = linear_model.LogisticRegression()
	lr.fit(X_train, y_train)
	y_pred_l = lr.predict(X_test)

	lscore = accuracy_score(y_test, y_pred_l)
	lacc.append(lscore)

	#trian random forest model
	rf = RandomForestClassifier()
	rf.fit(X_train, y_train)
	y_pred_2 = rf.predict(X_test)

	rscore = accuracy_score(y_test, y_pred_2)
	racc.append(rscore)

	#train svc
	svc = svm.SVC()
	svc.fit(X_train, y_train)
	y_pred_3 = svc.predict(X_test)

	sscore = accuracy_score(y_test, y_pred_3)
	sacc.append(sscore)

	#k-fold cv
	kfold = KFold(n_splits=5, random_state=5)
	kfscore_xgb = cross_val_score(xc, X_train, y_train, cv=kfold)
	kfold = KFold(n_splits=5, random_state=5)
	kfscore_lr = cross_val_score(lr, X_train, y_train, cv=kfold)
	kfold = KFold(n_splits=5, random_state=5)
	kfscore_rf = cross_val_score(rf, X_train, y_train, cv=kfold)
	kfold = KFold(n_splits=5, random_state=5)
	kfscore_svc = cross_val_score(svc, X_train, y_train, cv=kfold)

	#simulate matches and store results
	winners, place = simulate_matches(teams, svc)
	place.insert(0, file)
	place.append(np.mean(lacc))
	place.append(np.mean(racc))
	place.append(np.mean(sacc))
	place.append(np.mean(xacc))
	place.append(round(kfscore_xgb.mean()*100, 2))
	place.append(round(kfscore_lr.mean()*100, 2))
	place.append(round(kfscore_rf.mean()*100, 2))
	place.append(round(kfscore_svc.mean()*100, 2))
	places.append(place)
#tour_table = pd.DataFrame(results, columns= ['Season', 'Round 1', 'Round 2', 'Round 3', 'Final', 'LR', 'RFC', 'SVC', 'XGB', 'Validation Score'])
#tour_table.to_csv('Data/Prediction/tournament_prediction.csv', index=False)
final_result = pd.DataFrame(places, columns=['Season', '5th', '4th', '3rd', '2nd', 'WINNER', 'LR', 'RFC', 'SVC', 'XGB', 'X_Val', 'L_Val', 'R_Val', 'S_Val'])
final_result.to_csv('Data/Prediction/' + league + '_final_standings.csv', index=False)