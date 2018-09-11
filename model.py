import pandas as pd
import os
from sklearn import linear_model, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

files = []
for file in os.listdir('C:/Users/khyu7/Documents/Coding/Lol/Data'):
	file_name, file_ext = os.path.splitext(file)
	if file_ext == '.csv' and 'Playoffs' not in file_name and 'Standings' not in file_name:
		files.append(file_name)
print(files)

def match(data, team1, team2, model, random_scale=5):
	match = pd.DataFrame(columns = ['sp1', 'gp1', 'p1', 'sp2', 'gp2', 'p2'], index=[0])
		
	match['sp1'] = data[data.Team == team1]['SPF'].iloc[0]
	match['gp1'] = data[data.Team == team1]['GPF'].iloc[0]
	match['p1'] = data[data.Team == team1]['P'].iloc[0]
	match['sp2'] = data[data.Team == team2]['SPF'].iloc[0]
	match['sp2'] = -match['sp2']
	match['gp2'] = data[data.Team == team2]['GPF'].iloc[0]
	match['gp2'] = -match['gp2']
	match['p2'] = data[data.Team == team2]['P'].iloc[0]
	match['p2'] = -match['p2']
	
	match_array = match.values
	
	prediction = model.predict(match_array)
	
	winner = None
	
	if prediction == 1:
		winner = team1
	elif prediction == 0:
		winner = team2
	return winner

def simulate_matches(team, model, n_matches=100):
	print(team)
	winner = team[0]
	print('-------------------------')
	print()
	for i in range(1,5):
		match_results = []
		for j in range(n_matches):
			match_results.append(match(standings, winner, team[i], model, random_scale=5))

		team1_proba = match_results.count(winner)/len(match_results)*100
		team2_proba = match_results.count(team[i])/len(match_results)*100

		print(winner, ': ', team1_proba)
		print(team[i], ': ', team2_proba)
		print()
		print('-------------------------')
		print()

		if team1_proba > team2_proba:
			winner = winner
		else:
			winner = team[i]
			
	return winner

lacc =[]
racc = []
sacc = []
#read files
file = 'LCK Summer 2016'

regular = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + '.csv', index_col=0)
playoff = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Playoffs.csv', index_col=0)
standings = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Standings.csv', index_col=0)

teams = np.unique(playoff[['Blue', 'Red']].values)

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

svc = svm.SVC(kernel='linear')
svc.fit(train_X, train_Y)
y_pred_s = svc.predict(test_X)

lscore = accuracy_score(test_Y, y_pred_l)
lacc.append(lscore)

test_Y = pd.DataFrame(test_Y)
playoff['lr'] = y_pred_l
playoff['svc'] = y_pred_s
#print(playoff.iloc[:, 3:])
#print(standings)
'''
match_results = []
for i in range(100000):
	match_results.append(match(standings, match(standings, match(standings, match(standings, 'Afreeca Freecs', 'Samsung Galaxy', svc), 'SK Telecom T1', svc), 'KT Rolster', svc), 'ROX Tigers', svc))
'''
final_winner = simulate_matches(teams, lr)
print('Winner of the tournament: ', final_winner)

print(np.mean(lacc))
#playoff.to_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/Prediction/' + file + '_pred.csv')