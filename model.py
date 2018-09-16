import pandas as pd
import os
from sklearn import linear_model, preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import xgboost as xgb

#set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('Data/Prediction'):
    os.mkdir('Data/Prediction')

def match(data, team1, team2, model, random_scale=5):
	match = pd.DataFrame(columns = ['sp1', 'gp1', 'p1', 'sp2', 'gp2', 'p2'], index=[0])

	gd1 = data[data.Team == team1]['GD'].iloc[0]
	kd1 = data[data.Team == team1]['KD'].iloc[0]
	td1 = data[data.Team == team1]['TD'].iloc[0]
	dd1 = data[data.Team == team1]['DD'].iloc[0]
	bd1 = data[data.Team == team1]['BD'].iloc[0]
	rhd1 = data[data.Team == team1]['RHD'].iloc[0]

	match['sp1'] = data[data.Team == team1]['SPF'].iloc[0]
	match['gp1'] = data[data.Team == team1]['GPF'].iloc[0]
	match['p1'] = data[data.Team == team1]['P'].iloc[0]
	match['sp2'] = data[data.Team == team2]['SPF'].iloc[0]
	match['sp2'] = -match['sp2']
	match['gp2'] = data[data.Team == team2]['GPF'].iloc[0]
	match['gp2'] = -match['gp2']
	match['p2'] = data[data.Team == team2]['P'].iloc[0]
	match['p2'] = -match['p2']

	match['gd1'] = np.random.normal(gd1, scale=random_scale)
	match['kd1'] = np.random.normal(kd1, scale=random_scale)
	match['td1'] = np.random.normal(td1, scale=random_scale)
	match['dd1'] = np.random.normal(dd1, scale=random_scale)
	match['bd1'] = np.random.normal(bd1, scale=random_scale)
	match['rhd1'] = np.random.normal(rhd1, scale=random_scale)
	
	match_array = match.values
	
	prediction = model.predict(match_array)
	
	winner = None
	
	if prediction == 1:
		winner = team1
	elif prediction == 0:
		winner = team2
	return winner

def simulate_matches(team, model, n_matches=3500):
	winner = team[0]
	print('-------------------------')
	print()
	winners = []
	for i in range(1,5):
		match_results = []
		for j in range(n_matches):
			match_results.append(match(standings, winner, team[i], model, random_scale=5))
			match_results.append(match(standings, team[i], winner, model, random_scale=5))

		team1_proba = match_results.count(winner)/len(match_results)*100
		team2_proba = match_results.count(team[i])/len(match_results)*100

		print(winner, ': ', team1_proba)
		print(team[i], ': ', team2_proba)

		if team1_proba > team2_proba:
			winner = winner
		else:
			winner = team[i]
		winners.append(winner)
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
	return winners

#read files
path = 'Data/'
files = []
for file in os.listdir(path):
	file_name, file_ext = os.path.splitext(file)
	if file_ext == '.csv' and 'Playoffs' not in file_name and 'Standings' not in file_name:
		files.append(file_name)

results = []
lacc = []
racc = []
sacc = []
time = []
results = []
lacc = []
racc = []
sacc = []
xacc = []
for file in files:
	print(file)
	regular = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + '.csv', index_col=0)
	playoff = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Playoffs.csv', index_col=0)
	standings = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Standings.csv', index_col=0)
	standings = standings.iloc[0:5]

	teams = standings['Team'].iloc[::-1]
	teams = teams.reset_index(drop=True)

	#xgboost testing
	X = regular.iloc[:, 3:-1]
	y = regular.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(X_train)
	X_train = pd.DataFrame(np_scaled)
	np_test_scaled = min_max_scaler.fit_transform(X_test)
	X_test = pd.DataFrame(np_test_scaled)

	model = xgb.XGBClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	xacc.append(accuracy)

	lr = linear_model.LogisticRegression()
	lr.fit(X_train, y_train)
	y_pred_l = lr.predict(X_test)

	lscore = accuracy_score(y_test, y_pred_l)
	lacc.append(lscore)

	rf = RandomForestClassifier()
	rf.fit(X_train, y_train)
	y_pred_2 = rf.predict(X_test)

	rscore = accuracy_score(y_test, y_pred_2)
	racc.append(rscore)

	svc = svm.SVC()
	svc.fit(X_train, y_train)
	y_pred_3 = svc.predict(X_test)

	sscore = accuracy_score(y_test, y_pred_3)
	sacc.append(sscore)

	winners = simulate_matches(teams, svc)
	winners.insert(0, file)
	winners.append(np.mean(lacc))
	winners.append(np.mean(racc))
	winners.append(np.mean(sacc))
	winners.append(np.mean(xacc))
	results.append(winners)
tour_table = pd.DataFrame(results, columns= ['Season', 'Round 1', 'Round 2', 'Round 3', 'Final', 'LR', 'RFC', 'SVC', 'XGB'])
tour_table.to_csv('Data/Prediction/tournament_prediction.csv', index=False)