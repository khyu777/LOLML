import pandas as pd
import os
from sklearn import linear_model, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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
for file in files:
	print(file)
	regular = pd.read_csv(path + file + '.csv', index_col=0)
	playoff = pd.read_csv(path + file + ' Playoffs.csv', index_col=0)
	standings = pd.read_csv(path + file + ' Standings.csv', index_col=0)
	standings = standings.iloc[0:5]

	teams = standings['Team'].iloc[::-1]
	teams = teams.reset_index(drop=True)

	train_X = regular.iloc[:, 3:-1]
	train_Y = regular.iloc[:, -1]
	test_X = playoff.iloc[:, 3:-1]
	test_Y = playoff.iloc[:, -1]
	'''
	train_X = (train_X-train_X.mean())/train_X.std()
	test_X = (test_X-test_X.mean())/test_X.std()
	'''

	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(train_X)
	train_X = pd.DataFrame(np_scaled)

	np_test_scaled = min_max_scaler.fit_transform(test_X)
	test_X = pd.DataFrame(np_test_scaled)

	lr = linear_model.LogisticRegression()
	lr.fit(train_X, train_Y)
	y_pred_l = lr.predict(test_X)

	lscore = accuracy_score(test_Y, y_pred_l)
	lacc.append(lscore)

	rf = RandomForestClassifier()
	rf.fit(train_X, train_Y)
	y_pred_2 = rf.predict(test_X)

	rscore = accuracy_score(test_Y, y_pred_2)
	racc.append(rscore)

	svc = svm.SVC()
	svc.fit(train_X, train_Y)
	y_pred_3 = svc.predict(test_X)

	sscore = accuracy_score(test_Y, y_pred_3)
	sacc.append(sscore)
	startTime = datetime.now()
	winners = simulate_matches(teams, rf)
	time.append(datetime.now()-startTime)
	winners.insert(0, file)
	winners.append(np.mean(lacc))
	winners.append(np.mean(racc))
	winners.append(np.mean(sacc))
	results.append(winners)
print(np.sum(time))
tour_table = pd.DataFrame(results, columns= ['Season', 'Round 1', 'Round 2', 'Round 3', 'Final', 'LR', 'RFC', 'SVC'])
tour_table.to_csv(path + 'Prediction/tournament_prediction.csv', index=False)