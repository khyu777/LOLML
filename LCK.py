#import libraries
import urllib.request as urllib
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#specify the url
lck = 'https://lol.gamepedia.com/Portal:Tournaments/South_Korea'

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
    if x == ' - ':
        x = 0
    return x

#get original url
def geturl():
    global teams
    global standings
    fin = soup.find('a', text=i, href=True)
    url = base + str(fin['href'])
    data = urllib.urlopen(url)
    print('URL read')
    title = BeautifulSoup(data, 'lxml')
    print('BS read')
    teams = {}
    for j in title.find_all('th'):
        if 'Season Standings' in j.text:
            standings = getstg(j)
            results, history = getmh(title)
            teams = gettm(results)
            stats = getms(history)
            output(results, stats)

#get standings
def getstg(j):
    standings = pd.read_html(str(j.find_parent('table')))
    standings = standings[0].iloc[8:, 1:].reset_index(drop=True)
    standings.columns = ['Team', 'Series', 'SP', 'Games', 'GP', 'P']
    standings['SPF'] = standings.SP.apply(p2f)
    standings['GPF'] = standings.GP.apply(p2f)
    standings['P'] = pd.to_numeric(standings['P'])
    return standings

#get match history
def getmh(title, x=''):
    name = title.find('h1', {'id': 'firstHeading'}).text
    name = name.replace(" ", "%20")

    if x == '':
        url2 = beg + str(name) + end
    else:
        url2 = beg + str(name) + x + end
    match = BeautifulSoup(urllib.urlopen(url2), 'lxml')
    print('BS Read...')
    history = match.find('table', {'class': 'wikitable'})
    print('Table found...')
    mh = pd.read_html(str(history), header=1)            
    results = mh[0].iloc[:, 2:5].dropna()            
    results['Blue'] = results['Blue'].str.replace('e-mFire', 'Kongdoo Monster')
    results['Red'] = results['Red'].str.replace('e-mFire', 'Kongdoo Monster')

    return results, history

#get teams
def gettm(results):
    tns = standings['Team'].sort_values()
    unq = results['Blue'].drop_duplicates().sort_values()
    
    teams = dict(zip(tns, unq))
    print(teams)
    standings['Team'] = standings['Team'].map(teams)
    print(standings)
    #standings.to_csv("C:/Users/khyu7/Documents/Coding/LOL/Data/" + i + ' Standings.csv')

    return teams

#get match stats
def getms(history):
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

#get team stats & combine with match stats, save to csv
def output(results, stats, x = ''):
    results['spf_b'] = results['Blue'].map(standings.set_index('Team')['SPF'])
    results['gpf_b'] = results['Blue'].map(standings.set_index('Team')['GPF'])
    results['spf_r'] = results['Red'].map(standings.set_index('Team')['SPF'])
    results['spf_r'] = -results['spf_r']
    results['gpf_r'] = results['Red'].map(standings.set_index('Team')['GPF'])
    results['gpf_r'] = -results['gpf_r']
    results['p_b'] = results['Blue'].map(standings.set_index('Team')['P'])
    results['p_r'] = results['Red'].map(standings.set_index('Team')['P'])
    results['p_r'] = -results['p_r']
    results['Winner'] = np.where(results['Win'] == 'blue', 1, 0)
    results = pd.concat([results,stats], axis=1)
    print(results)
    print(standings)
    path = "C:/Users/khyu7/Documents/Coding/LOL/Data/"
    #if x == '':
        #results.to_csv(path + i + '.csv')
    #else:
        #results.to_csv(path + i + x + '.csv')

def getmean(results):
    for team in standings['Team'].iloc[0:5]:
        diff = results.loc[results['Blue'] == team].iloc[:, -6:]
        diff = diff.append(-results.loc[results['Red'] == team].iloc[:, -6:])
        diff = diff.reset_index(drop=True)
        diff.loc['Mean'] = diff.mean()
        print(team)
        print(diff)

#get unique tournament list & url
base = 'https://lol.gamepedia.com'
beg = 'https://lol.gamepedia.com/Special:RunQuery/MatchHistoryTournament?MHT%5Btournament%5D=Concept:'
end = '&MHT%5Blimit%5D=250&MHT%5Boffset%5D=0&MHT%5Btext%5D=Yes&pfRunQueryFormName=MatchHistoryTournament'
teams = {}
for i in df[0]['Tournament']:
    if 'LCK' in i:
        print(i)
        geturl()
        print()
        print(i + ' Playoffs')
        fin = soup.find('a', text=i, href=True)
        url = base + str(fin['href'])
        data1 = urllib.urlopen(url)
        print('URL Read')
        title = BeautifulSoup(data1, 'lxml')
        print('BS Read')
        results, history = getmh(title, x='%20Playoffs')
        stats = getms(history)
        output(results, stats, x = ' Playoffs')