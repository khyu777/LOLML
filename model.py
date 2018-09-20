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
def match(data, data2, team1, team2, model, random_scale=1):
	match = pd.DataFrame(columns = ['sp1', 'gp1', 'p1', 'sp2', 'gp2', 'p2'], index=[0])

	match['sp1'] = data[data.Team == team1]['SPF'].iloc[0]
	match['sp2'] = data[data.Team == team2]['SPF'].iloc[0]
	match['spf_diff'] = match['sp1'] - match['sp2']
	match = match.drop(['sp1', 'sp2'], axis=1)
	if len(data.columns) == 13 or len(data.columns) == 7:
		match['gp1'] = data[data.Team == team1]['GPF'].iloc[0]
		match['gp2'] = data[data.Team == team2]['GPF'].iloc[0]
		match['gpf_diff'] = match['gp1'] - match['gp2']
		match = match.drop(['gp1', 'gp2'], axis=1)
	elif len(data.columns) == 14 or len(data.columns) == 8:
		match['gp1'] = data[data.Team == team1]['GPF'].iloc[0]
		match['gp2'] = data[data.Team == team2]['GPF'].iloc[0]
		match['gpf_diff'] = match['gp1'] - match['gp2']
		match['p1'] = data[data.Team == team1]['P'].iloc[0]
		match['p2'] = data[data.Team == team2]['P'].iloc[0]
		match['p_diff'] = match['p1'] - match['p2']
		match = match.drop(['gp1', 'gp2', 'p1', 'p2'], axis=1)

	gd1 = data2[(data2.Team == team1)&(data2.Against == team2)]['GD'].iloc[0]
	kd1 = data2[(data2.Team == team1)&(data2.Against == team2)]['KD'].iloc[0]
	td1 = data2[(data2.Team == team1)&(data2.Against == team2)]['TD'].iloc[0]
	dd1 = data2[(data2.Team == team1)&(data2.Against == team2)]['DD'].iloc[0]
	bd1 = data2[(data2.Team == team1)&(data2.Against == team2)]['BD'].iloc[0]
	rhd1 = data2[(data2.Team == team1)&(data2.Against == team2)]['RHD'].iloc[0]

	match['GD'] = np.random.normal(gd1)
	match['KD'] = np.random.normal(kd1)
	match['TD'] = np.random.normal(td1)
	match['DD'] = np.random.normal(dd1)
	match['BD'] = np.random.normal(bd1)
	match['RHD'] = np.random.normal(rhd1)
	
	match = match.dropna(axis=1)
	print('match')
	print(match)
	
	return match

def match_run(predict, team1, team2):
	choices = [0, 1]
	prediction = random.choices(choices, predict[0])
	winner = None
	
	if prediction[0] == 1:
		winner = team1
	elif prediction[0] == 0:
		winner = team2
	return winner

#match simulation
def simulate_matches(team, model, n_matches=6000):
	winner = team[0]
	print('-------------------------')
	print()
	winners = []
	place = []
	for i in range(1,5):
		match_results = []
		match_array = match(standings, jj, winner, team[i], model)
		predict = model.predict_proba(match_array)
		for j in range(n_matches):
			match_results.append(match_run(predict, winner, team[i]))

		team1_proba = match_results.count(winner)/len(match_results)*100
		team2_proba = match_results.count(team[i])/len(match_results)*100

		print(winner, ': ', team1_proba)
		print(team[i], ': ', team2_proba)

		if team1_proba > team2_proba:
			winner = winner
			loser = team[i]
		elif team1_proba < team2_proba:
			loser = winner
			winner = team[i]
		winners.append(winner)
		if i < 5:
			place.append(loser)
		'''
		p_list = []
		for num in range(len(match_results)):
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
a = ['Standings', '_jeonjuk', 'Playoffs']
for file in os.listdir(path):
	file_name, file_ext = os.path.splitext(file)
	if file_ext == '.csv' and not any(x in file_name for x in a):
		files.append(file_name)
print(files)
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
	regular = pd.read_csv(path + file + '.csv', index_col=0)
	jj = pd.read_csv(path + file + '_jeonjuk.csv', index_col=0)
	standings = pd.read_csv(path + file + ' Standings.csv', index_col=0)
	standings = standings.iloc[0:5]

	teams = standings['Team'].iloc[::-1]
	teams = teams.reset_index(drop=True)

	#xgboost testing
	X = regular.iloc[:, 3:-1]
	y = regular.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	'''
	#normalize data
	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(X_train)
	X_train = pd.DataFrame(np_scaled).values
	np_test_scaled = min_max_scaler.fit_transform(X_test)
	X_test = pd.DataFrame(np_test_scaled).values
	'''
	#train XGB model
	xc = xgb.XGBClassifier(probability=True)
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
	svc = svm.SVC(probability=True)
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
	winners, place = simulate_matches(teams, xc)
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
final_result = pd.DataFrame(places, columns=['Season', '5th', '4th', '3rd', '2nd', 'WINNER', 'LR', 'RFC', 'SVC', 'XGB', 'X_Val', 'L_Val', 'R_Val', 'S_Val'])
final_result.to_csv('Data/Prediction/' + league + '_final_standings.csv', index=False)