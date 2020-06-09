import json

def draw(name, data):
    x = data[0]
    y = data[1]
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    line, = ax.plot(x, y, '-')

    ax.grid()
    # ax.axis([1,5,1,5])
    ax.axis('equal')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title(f'{name} VA curve')
    plt.tight_layout()
    plt.savefig('VAcurveOut/' + name+'.png')
    plt.close('all')
    

with open('predictVA.json', 'r') as f:
    VA = json.load(f)
    for name, curve in VA.items():
        draw(name, curve)

    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.subplot(131)
    plt.plot(VA['Wanted.mp4'][0], VA['Wanted.mp4'][1], '-')
    plt.axis('equal')
    plt.subplot(132)
    plt.plot(VA['Chatter.mp4'][0], VA['Chatter.mp4'][1], '-')
    plt.axis('equal')
    plt.subplot(133)
    plt.plot(VA['Elephant_s_Dream.mp4'][0], VA['Elephant_s_Dream.mp4'][1], '-')
    plt.axis('equal')
    fig, ax = plt.subplots()
    
    # ax.plot(VA['Wanted.mp4'][0], VA['Wanted.mp4'][1], '-', label='Wanted')
    ax.plot(VA['Chatter.mp4'][0], VA['Chatter.mp4'][1], '-', label='Chatter')
    ax.plot(VA['Elephant_s_Dream.mp4'][0], VA['Elephant_s_Dream.mp4'][1], '-', label="Elephan's Dream")
    ax.grid()
    # ax.axis([1,5,1,5])
    ax.axis('equal')
    ax.legend()
    plt.xlabel('Arousal')
    plt.ylabel('Valence')
    plt.tight_layout()
    plt.savefig('VAcurveOut/VAcurve.png')