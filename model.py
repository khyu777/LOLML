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

def simulate_matches(team1, team2, n_matches=10000):
    
      match_results = []
      for i in range(n_matches):
            match_results.append(match(standings, team1, team2, svc, random_scale=5))
            
      team1_proba = match_results.count(team1)/len(match_results)*100
      team2_proba = match_results.count(team2)/len(match_results)*100
      
      print(team1, str(round(team1_proba, 2)) + '%')
      print(team2, str(round(team2_proba,2)) + '%')
      print('-------------------------')
      print()
      
      if team1_proba > team2_proba:
            overall_winner = team1
      else:
            overall_winner = team2
      
      return {'team1': team1,
                  'team2': team2,
                  'team1_proba': team1_proba, 
                  'team2_proba': team2_proba, 
                  'overall_winner': overall_winner,
                  'match_results': match_results}

lacc =[]
racc = []
sacc = []

file = 'LCK Summer 2016'

regular = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + '.csv', index_col=0)
playoff = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Playoffs.csv', index_col=0)
standings = pd.read_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + file + ' Standings.csv', index_col=0)

teams = np.unique(playoff[['Blue', 'Red']].values)

train_X = regular.iloc[:, 3:-1]
train_Y = regular.iloc[:, -1]
test_X = playoff.iloc[:, 3:-1]
test_Y = playoff.iloc[:, -1]

train_X = (train_X-train_X.mean())/train_X.std()
test_X = (test_X-test_X.mean())/test_X.std()

'''
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train_X)
train_X = pd.DataFrame(np_scaled)
print(train_X)

np_test_scaled = min_max_scaler.fit_transform(test_X)
test_X = pd.DataFrame(np_test_scaled)
print(test_X)
'''
lr = linear_model.LogisticRegression()
lr.fit(train_X, train_Y)
y_pred_l = lr.predict(test_X)

rf = RandomForestClassifier()
rf.fit(train_X, train_Y)
y_pred_r = rf.predict(test_X)

svc = svm.SVC(kernel='linear')
svc.fit(train_X, train_Y)
y_pred_s = svc.predict(test_X)

lscore = accuracy_score(test_Y, y_pred_l)
lacc.append(lscore)
rscore = accuracy_score(test_Y, y_pred_r)
racc.append(rscore)
sscore = accuracy_score(test_Y, y_pred_s)
sacc.append(sscore)

test_Y = pd.DataFrame(test_Y)
playoff['lr'] = y_pred_l
playoff['rf'] = y_pred_r
playoff['svc'] = y_pred_s
print(playoff.iloc[:, 3:])
print(standings)

final_winner = match(standings, match(standings, match(standings, match(standings, 'Afreeca Freecs', 'Samsung Galaxy', svc), 'SK Telecom T1', svc), 'KT Rolster', svc), 'ROX Tigers', svc)
print(final_winner)

print(np.mean(lacc), np.mean(racc), np.mean(sacc))
#playoff.to_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/Prediction/' + file + '_pred.csv')