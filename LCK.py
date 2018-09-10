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

def getdata():
    global teams
    global standings
    fin = soup.find('a', text=i, href=True)
    url = base + str(fin['href'])
    title = BeautifulSoup(urllib.urlopen(url), 'html.parser')
    teams = {}
    for j in title.find_all('th'):
        if 'Season Standings' in j.text:
            standings = pd.read_html(str(j.find_parent('table')))
            standings = standings[0].iloc[8:, 1:].reset_index(drop=True)
            standings.columns = ['Team', 'Series', 'SP', 'Games', 'GP', 'P']
            standings['SPF'] = standings.SP.apply(p2f)
            standings['GPF'] = standings.GP.apply(p2f)
            standings['P'] = pd.to_numeric(standings['P'])
            standings.to_csv("C:/Users/khyu7/Documents/Coding/LOL/Data/" + i + ' Standings.csv')
            
            name = title.find('h1', {'id': 'firstHeading'}).text
            name = name.replace(" ", "%20")

            url2 = beg + str(name) + end
            match = BeautifulSoup(urllib.urlopen(url2), 'html.parser')
            history = match.find('table', {'class': 'wikitable'})
            mh = pd.read_html(str(history), header=1)            
            results = mh[0].iloc[:, 2:5].dropna()            
            results['Blue'] = results['Blue'].str.replace('e-mFire', 'Kongdoo Monster')
            results['Red'] = results['Red'].str.replace('e-mFire', 'Kongdoo Monster')

            tns = standings['Team'].sort_values()
            unq = results['Blue'].drop_duplicates().sort_values()
            
            teams = dict(zip(tns, unq))
            print(teams)
            standings['Team'] = standings['Team'].map(teams)
            print(standings)

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
            #results = results.iloc[:, 3:]
            #print(results)
            path = "C:/Users/khyu7/Documents/Coding/LOL/Data/"
            results.to_csv(path + i + '.csv')

#get unique tournament list & url
base = 'https://lol.gamepedia.com'
beg = 'https://lol.gamepedia.com/Special:RunQuery/MatchHistoryTournament?MHT%5Btournament%5D=Concept:'
end = '&MHT%5Blimit%5D=250&MHT%5Boffset%5D=0&MHT%5Btext%5D=Yes&pfRunQueryFormName=MatchHistoryTournament'
teams = {}
for i in df[0]['Tournament']:
    print(i)
    getdata()
    if 'LCK' in i:     
        fin = soup.find('a', text=i, href=True)
        url = base + str(fin['href'])
        title = BeautifulSoup(urllib.urlopen(url), 'html.parser')

        name = title.find('h1', {'id': 'firstHeading'}).text
        name = name.replace(" ", "%20")

        url2 = beg + str(name) + '%20Playoffs' + end
        match = BeautifulSoup(urllib.urlopen(url2), 'html.parser')
        history = match.find('table', {'class': 'wikitable'})
        mh = pd.read_html(str(history), header=1)            
        results = mh[0].iloc[:, 2:5].dropna()            
        results['Blue'] = results['Blue'].str.replace('e-mFire', 'Kongdoo Monster')
        results['Red'] = results['Red'].str.replace('e-mFire', 'Kongdoo Monster')

        tns = standings['Team'].sort_values()
        unq = results['Blue'].drop_duplicates().sort_values()
        
        #print(standings)
        
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
        #results = results.iloc[:, :]
        #print(results)
        path = "C:/Users/khyu7/Documents/Coding/LOL/Data/"
        results.to_csv(path + i + ' Playoffs' + '.csv')

#model = LinearRegression()
#model.fit(results.iloc[:,0:4], results.iloc[:,5])