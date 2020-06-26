from scipy import stats
import json
import statistics
import os


def drawCorrelation(names, values, correlation_name):
    import matplotlib.pyplot as plt
    import matplotlib
    import random

    matplotlib.rcParams['axes.unicode_minus'] = False
    name_value_pair = list(zip(names, values))
    def cmp(x):
        return -x[1]
    name_value_pair.sort(key=cmp)
    name_value_pair = name_value_pair[:20]
    random.shuffle(name_value_pair)
    names, values = zip(*name_value_pair)
    n = len(names)
    plt.barh(range(n), values, height=0.7,
             color='steelblue', alpha=0.8)      # 从下往上画
    plt.yticks(range(n), names)
    plt.xlim(-0.5, 0.5)
    plt.xlabel(correlation_name)
    plt.title(correlation_name)
    # plt.setp(plt.yticks()[1], rotation=30, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(correlation_name+'.png')
    print (correlation_name, statistics.mean(values))


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


with open('predictVA.json', 'r') as f:
    VA = json.load(f)
    correlation = {}
    for i in fpsMovie:
        movieName = i[0]
        model = VA[movieName+'.mp4']
        phy = getPhysiological(i[1])
        phyArousal = multi(phy['arousal'], 5)
        phyValence = multi(phy['valence'], 5)
        # modelArousal = [i * -500 for i in multi(model[1], 8)]
        # modelValence = [i * -500 for i in multi(model[0], 8)]
        modelArousal = multi(model[1], 8)
        modelValence = multi(model[0], 8)
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
    pearsonrArousal = []
    pearsonrValence = []
    spearmanrArousal = []
    spearmanrValence = []
    pointbiserialrArousal = []
    pointbiserialrValence = []
    kendalltauArousal = []
    kendalltauValence = []
    names = []
    for i, v in correlation.items():
        names.append(i)
        pearsonrArousal.append(v[0][0][0])
        pearsonrValence.append(v[0][1][0])
        spearmanrArousal.append(v[1][0][0])
        spearmanrValence.append(v[1][1][0])
        pointbiserialrArousal.append(v[2][0][0])
        pointbiserialrValence.append(v[2][1][0])
        kendalltauArousal.append(v[3][0][0])
        kendalltauValence.append(v[3][1][0])
    print('Pearson: ')
    print(statistics.mean(pearsonrArousal))
    drawCorrelation(names, pearsonrArousal, 'pearsonrArousal')
    print(statistics.mean(pearsonrValence))
    drawCorrelation(names, pearsonrValence, 'pearsonrValence')
    print('spearmanr: ')
    print(statistics.mean(spearmanrArousal))
    drawCorrelation(names, spearmanrArousal, 'spearmanrArousal')
    print(statistics.mean(spearmanrValence))
    drawCorrelation(names, spearmanrValence, 'spearmanrValence')
    print('pointbiserialr: ')
    print(statistics.mean(pointbiserialrArousal))
    drawCorrelation(names, pointbiserialrArousal, 'pointbiserialrArousal')
    print(statistics.mean(pointbiserialrValence))
    drawCorrelation(names, pointbiserialrValence, 'pointbiserialrValence')
    print('kendalltau: ')
    print(statistics.mean(kendalltauArousal))
    drawCorrelation(names, kendalltauArousal, 'kendalltauArousal')
    print(statistics.mean(kendalltauValence))
    drawCorrelation(names, kendalltauValence, 'kendalltauValence')
