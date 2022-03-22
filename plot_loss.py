import os
import argparse
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--input", type=str, help="path to the log")
arg_parser.add_argument("--output", type=str, help="directory to save loss curves")
args = arg_parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

def main():
    loss_dict = dict()
    for line in open(args.input):
        if "Epoch:" in line:
            continue
        else:
            info = line.rstrip().split(" ")
            prefix = info[0][1:-1]
            for i in range(1, len(info)):
                try:
                    value = float(info[i])
                except ValueError:
                    continue
                key = prefix + "_" + info[i-1].replace(":", "")
                if not key in loss_dict:
                    loss_dict[key] = list()
                loss_dict[key].append(value)

    for k, v in loss_dict.items():
        curve_path = os.path.join(args.output, k + ".png")
        """
        new_v = list()
        l = 10
        for i in range(l, len(v) - l):
            new_v.append(sum(v[i-l:i+l]) / (2 * l))
        """
        new_v = v
        plt.plot(list(range(len(new_v))), new_v)
        plt.title(k)
        plt.savefig(curve_path)
        plt.clf()



if __name__ == '__main__':
    main()

