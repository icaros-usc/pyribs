import matplotlib.pyplot as plt
import seaborn as sns

from ribs.archives import GridArchive
from ribs.optimizers import Optimizer

if __name__ == '__main__':

    archive = GridArchive((100, 100), [(-4, 4), (-4, 4)])
    opt = Optimizer([0.0] * 10, 0.1, 4, archive)

    for i in range(10**5):
        sols = opt.ask()
        objs = [sum(s) for s in sols]
        bcs = [(s[0], s[1]) for s in sols]

        opt.tell(sols, objs, bcs)

        if i % 1000 == 0:
            print('saving {}'.format(i))
            #data = opt.archive.as_pandas()
            #data = data.pivot('index-0', 'index-1', 'objective')

            #ax = sns.heatmap(data)
            #plt.savefig('images/arc-{:05d}'.format(i))
            #plt.close()
