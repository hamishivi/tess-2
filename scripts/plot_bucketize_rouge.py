import json 

data = json.load(open('our_model_cnn_bucket_results.json', 'r'))


import matplotlib.pyplot as plt

# Uncomment the following line if you use ipython notebook
# %matplotlib inline

width = 0.35       # the width of the bars

men_means = {'G1': 20, 'G2': 35, 'G3': 30, 'G4': 35, 'G5': 27}
men_std = {'G1': 2, 'G2': 3, 'G3': 4, 'G4': 1, 'G5': 2}

rects1 = plt.bar(men_means.keys(), men_means.values(), -width, align='edge',
                yerr=men_std.values(), color='r', label='Men')

women_means = {'G1': 25, 'G2': 32, 'G3': 34, 'G4': 20, 'G5': 25}
women_std = {'G1': 3, 'G2': 5, 'G3': 2, 'G4': 3, 'G5': 3}

rects2 = plt.bar(women_means.keys(), women_means.values(), +width, align='edge',
                yerr=women_std.values(), color='y', label='Women')

# add some text for labels, title and axes ticks
plt.xlabel('Groups')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.legend()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.savefig("figure.pdf")
plt.show()
