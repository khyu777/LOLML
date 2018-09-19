#import libraries
import urllib.request as urllib
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

#set working directory and create folders as necessary
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('Data'):
    os.mkdir('Data')

#specify the url
country = 'South_Korea'
lck = 'https://lol.gamepedia.com/Portal:Tournaments/' + country

#return html to variable 'page'
page = urllib.urlopen(lck)

#get tournament table
soup = BeautifulSoup(page, 'html.parser')
tournaments = soup.find('table', {"class" : "wikitable sortable"})
df = pd.read_html(str(tournaments), header=0)

def p2f(x):
    return float(x.strip('%'))/100

def g2int(x):
    x = x.strip('k')
    return float(x)

def tozero(x):
    if (x == ' - ') or (x == ' -'):
        x = 0
    return x

#get original url
def geturl():
    global teams
    global standings
    fin = soup.find('a', text=i, href=True)
    url = base + str(fin['href'])
    data = urllib.urlopen(url)
    title = BeautifulSoup(data, 'lxml')
    teams = {}
    for j in title.find_all('th'):
        if 'Season Standings' in j.text:
            standings = getstg(j)
            results = getmh(title)
            average = getmean(results, standings)
            output(results, standings, average)
    return title, standings

#get standings
def getstg(j):
    standings = pd.read_html(str(j.find_parent('table')))
    numeric = standings[0][0].str.isnumeric()
    teamcount = (numeric==True).sum()
    standings = standings[0].dropna(axis=1,thresh=teamcount)
    standings = standings.dropna()
    standings = standings.loc[standings[0].str.isnumeric()]
    standings = standings.iloc[:, 1:].reset_index(drop=True)
    if len(standings.columns) == 3:
        standings.columns = ['Team', 'Series', 'SP'] #, 'Games', 'GP', 'P']
    elif len(standings.columns) == 5:
        standings.columns = ['Team', 'Series', 'SP', 'Games', 'GP']
        standings['GPF'] = standings.GP.apply(p2f)
    elif len(standings.columns) == 6:
        standings.columns = ['Team', 'Series', 'SP', 'Games', 'GP', 'P']
        standings['GPF'] = standings.GP.apply(p2f)
        standings['P'] = pd.to_numeric(standings['P'])
    standings['SPF'] = standings.SP.apply(p2f)
    print(standings)
    return standings

#get match stats
def getms(history, results):
    print('match stats..')
    l = []
    for row in history.find_all('tr'):
        column = row.find_all('td', {'class':'_toggle stats'})
        values = []
        for value in column:
            values.append(value.text.strip('\n'))
        l.append(values)
    stats = pd.DataFrame(l).iloc[:,-6:]
    stats.columns = ['GD', 'KD', 'TD', 'DD', 'BD', 'RHD']
    stats = stats.dropna()
    stats = stats.reset_index(drop=True)
    stats['GD'] = stats.GD.apply(g2int)
    stats.RHD = stats.RHD.apply(tozero)
    stats.KD = pd.to_numeric(stats.KD)
    stats.TD = pd.to_numeric(stats.TD)
    stats.DD = pd.to_numeric(stats.DD)
    stats.BD = pd.to_numeric(stats.BD)
    stats.RHD = pd.to_numeric(stats.RHD)

    return stats

#get match history
def getmh(title, x=''):
    print('match history...')
    name = title.find('h1', {'id': 'firstHeading'}).text
    name = name.replace(" ", "%20")

    if x == '':
        url2 = beg + str(name) + end
    else:
        url2 = beg + str(name) + x + end
    match = BeautifulSoup(urllib.urlopen(url2), 'lxml')
    history = match.find('table', {'class': 'wikitable'})
    mh = pd.read_html(str(history), header=1)            
    results = mh[0].iloc[:, 2:5].dropna()            
    results['Blue'] = results['Blue'].str.replace('e-mFire', 'Kongdoo Monster')
    results['Red'] = results['Red'].str.replace('e-mFire', 'Kongdoo Monster')
    tns = standings['Team'].sort_values()
    unq = results['Blue'].drop_duplicates().sort_values()
    
    teams = dict(zip(tns, unq))
    if x == '':
        standings['Team'] = standings['Team'].map(teams)
    results['spf_b'] = results['Blue'].map(standings.set_index('Team')['SPF'])
    results['spf_r'] = results['Red'].map(standings.set_index('Team')['SPF'])
    results['spf_r'] = -results['spf_r']
    if len(standings.columns) == 7:
        results['gpf_b'] = results['Blue'].map(standings.set_index('Team')['GPF'])
        results['gpf_r'] = results['Red'].map(standings.set_index('Team')['GPF'])
        results['gpf_r'] = -results['gpf_r']
    elif len(standings.columns) == 8:
        results['gpf_b'] = results['Blue'].map(standings.set_index('Team')['GPF'])
        results['gpf_r'] = results['Red'].map(standings.set_index('Team')['GPF'])
        results['gpf_r'] = -results['gpf_r']
        results['p_b'] = results['Blue'].map(standings.set_index('Team')['P'])
        results['p_r'] = results['Red'].map(standings.set_index('Team')['P'])
        results['p_r'] = -results['p_r']

    stats = getms(history, results)
    results = pd.concat([results, stats], axis=1)
    results['Winner'] = np.where(results['Win'] == 'blue', 1, 0)

    return results

def getmean(results, standings):
    print('mean..')
    topfive = standings.iloc[:5, :]['Team']
    average = pd.DataFrame(columns=['Team', 'Against', 'GD', 'KD', 'TD', 'DD', 'BD', 'RHD'])
    for team in topfive:
        indiv = results.loc[results['Blue'] == team]
        indivi = results.loc[results['Red'] == team]
        for column in indivi.columns:
            if np.issubdtype(indivi[column], np.number):
                indivi[column] = -indivi[column]
        for j in topfive:
            if j != team:
                individ = pd.DataFrame()
                individ = individ.append(indiv.loc[indiv['Red'] == j])
                individ = individ.append(indivi.loc[indivi['Blue'] == j])
                individ = individ.iloc[:, -7:-1]
                individ.loc['Mean'] = individ.mean()
                s = individ.loc['Mean']
                s.loc['Against'] = j
                s.loc['Team'] = team
                average = average.append(s)
    average = average.reset_index(drop=True)
    return average

#get team stats & combine with match stats, save to csv
def output(results, standings, average, x = ''):
    print('output...')
    #standings = pd.concat([standings, average], axis=1)
    path = "Data/"
    if x == '':
    #    results.to_csv(path + i + '.csv')
        print(standings)
        print(average)
        #average.to_csv(path + i + '_jeonjuk.csv')
    #else:
    #    results.to_csv(path + i + x + '.csv')

#get unique tournament list & url
base = 'https://lol.gamepedia.com'
beg = 'https://lol.gamepedia.com/Special:RunQuery/MatchHistoryTournament?MHT%5Btournament%5D=Concept:'
end = '&MHT%5Blimit%5D=250&MHT%5Boffset%5D=0&MHT%5Btext%5D=Yes&pfRunQueryFormName=MatchHistoryTournament'
teams = {}
for i in df[0]['Tournament']:
    print(i)
    title, average = geturl()
    #print()
    #print(i + ' Playoffs')
    #results = getmh(title, x='%20Playoffs')
    #output(results, standings, average, x = ' Playoffs')