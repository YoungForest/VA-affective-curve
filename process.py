import nvidia.dali.ops as ops
def getVideoList():
    data_dir = '../LIRIS-ACCEDE-data/data/'

    import glob
    video_list = glob.glob(data_dir + '*.mp4')
    return video_list

def getRating():
    ranking_file = '../LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt'
    import pandas as pd
    ranking = pd.read_csv(ranking_file, delimiter='\t')
    return ranking

if __name__ == "__main__":
    video_list = getVideoList()
    print (video_list)

    print(getRating())
