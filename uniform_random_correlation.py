from scipy import stats
import json
import statistics
import os
import random

fpsMovie = [['Attitude_Matters', 'Attitude Matters'],
            ['Barely_legal_stories', 'Barely Legal Stories'],
            ['Between_Viewings', 'Between Viewings'],
            ['Big_Buck_Bunny', 'Big Buck Bunny'],
            ['Chatter', 'Chatter'],
            ['Cloudland', 'Cloudland'],
            ['Damaged_Kung_Fu', 'Damaged Kung-Fu'],
            ['Decay', 'Decay'],
            ['Elephant_s_Dream', 'Elephants Dream'],
            ['First_Bite', 'First Bite'],
            ['Full_Service', 'Full Service'],
            ['Islands', 'Islands'],
            ['Lesson_Learned', 'Lesson Learned'],
            ['Norm', 'Norm'],
            ['Nuclear_Family', 'Nuclear Family'],
            ['On_time', 'On Time'],
            ['Origami', 'Origami'],
            ['Parafundit', 'Parafundit'],
            ['Payload', 'Payload'],
            ['Riding_The_Rails', 'Riding the Rails'],
            ['Sintel', 'Sintel'],
            ['Spaceman', 'Spaceman'],
            ['Superhero', 'Superhero'],
            ['Tears_of_Steel', 'Tears of steel'],
            ['The_room_of_franz_kafka', 'The room of Franz Kafka'],
            ['The_secret_number', 'The secret number'],
            ['To_Claire_From_Sonny', 'To Claire From Sonny'],
            ['Wanted', 'Wanted'],
            ['You_Again', 'You Again']]


def getPhysiological(name):
    physiological_folder = '/root/yangsen-data/LIRIS-ACCEDE-continuous/physiological/continuous-physiological/'
    file = os.path.join(physiological_folder, name + '.csv')
    import pandas as pd
    df = pd.read_csv(file, sep=';', header=None, names=[
                     'time', 'arousal', 'valence'])
    return df


def multi(serial, times):
    ans = []
    for i in serial:
        ans += [i] * times
    return ans

def uniformSeed():
    return (1.0 + 5.0) / 2.0
def randomSeed():
    return 

correlation = {}
for i in fpsMovie:
    movieName = i[0]
    phy = getPhysiological(i[1])
    phyArousal = multi(phy['arousal'], 5)
    phyValence = multi(phy['valence'], 5)
    length = len(phyValence)
    assert(len(phyValence) == len(phyArousal))
    modelArousal = generate(length, seed)
    modelValence = generate(length, seed)
    minArousal = min(len(phyArousal), len(modelArousal))
    for _ in range(minArousal, len(phyArousal)):
        phyArousal.pop()
    for _ in range(minArousal, len(modelArousal)):
        modelArousal.pop()
    minValence = min(len(phyValence), len(modelValence))
    for _ in range(minValence, len(phyValence)):
        phyValence.pop()
    for _ in range(minValence, len(modelValence)):
        modelValence.pop()
    correlation[movieName] = [[stats.pearsonr(phyArousal, modelArousal), stats.pearsonr(phyValence, modelValence)],
                                [stats.spearmanr(phyArousal, modelArousal), stats.spearmanr(
                                    phyValence, modelValence)],
                                [stats.pointbiserialr(phyArousal, modelArousal),
                                stats.pointbiserialr(phyValence, modelValence)],
                                [stats.kendalltau(phyArousal, modelArousal), stats.kendalltau(phyValence, modelValence)]]
pearsonrAroual = []
pearsonrValence = []
spearmanrAroual = []
spearmanrValence = []
pointbiserialrAroual = []
pointbiserialrValence = []
kendalltauAroual = []
kendalltauValence = []
names = []
for i, v in correlation.items():
    names.append(i)
    pearsonrAroual.append(v[0][0][0])
    pearsonrValence.append(v[0][1][0])
    spearmanrAroual.append(v[1][0][0])
    spearmanrValence.append(v[1][1][0])
    pointbiserialrAroual.append(v[2][0][0])
    pointbiserialrValence.append(v[2][1][0])
    kendalltauAroual.append(v[3][0][0])
    kendalltauValence.append(v[3][1][0])
print('Pearson: ')
print(statistics.mean(pearsonrAroual))
drawCorrelation(names, pearsonrAroual, 'pearsonrAroual')
print(statistics.mean(pearsonrValence))
drawCorrelation(names, pearsonrValence, 'pearsonrValence')
print('spearmanr: ')
print(statistics.mean(spearmanrAroual))
drawCorrelation(names, spearmanrAroual, 'spearmanrAroual')
print(statistics.mean(spearmanrValence))
drawCorrelation(names, spearmanrValence, 'spearmanrValence')
print('pointbiserialr: ')
print(statistics.mean(pointbiserialrAroual))
drawCorrelation(names, pointbiserialrAroual, 'pointbiserialrAroual')
print(statistics.mean(pointbiserialrValence))
drawCorrelation(names, pointbiserialrValence, 'pointbiserialrValence')
print('kendalltau: ')
print(statistics.mean(kendalltauAroual))
drawCorrelation(names, kendalltauAroual, 'kendalltauAroual')
print(statistics.mean(kendalltauValence))
drawCorrelation(names, kendalltauValence, 'kendalltauValence')
