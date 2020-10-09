from ribs.optimizers import Optimizer

if __name__ == '__main__':

    opt = Optimizer([0.0] * 5, 1.0, 4)

    for i in range(10):
        sols = opt.ask()
        objs = [sum(s) for s in sols]
        bcs = [(s[0], s[1]) for s in sols]

        print(sols)

        opt.tell(sols, objs, bcs)
