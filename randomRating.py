def uniformRandomModel():
    rankingPd = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt', sep='\t', index_col='id')
    setsPd = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt', sep='\t', index_col='id')
    
