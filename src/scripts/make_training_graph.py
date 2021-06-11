"""Processes the experiment result and make it into a good visualization.

Usage: python3 -m src.scripts.make_training_graph results/training_result.txt results/training_result.png
"""
import matplotlib.pyplot as plt
import sys

text_path = sys.argv[1]
pic_path = sys.argv[2]


# Extract progress
import re
patterns = [r"Dev Accuracy : 0.([0-9]*)", r"Attack Accuracy : 0.([0-9]*)", r"Attack Accuracy \(Strategy 2\) : 0.([0-9]*)"]
dev_accs = [[], [], []]
f = open(text_path, "r")
for line in f:
    for i in range(len(dev_accs)):
        result = re.search(patterns[i], line)
        if result and (result.groups()[0]):
            dev_accs[i].append(float('0.' + result.groups()[0]))

print(dev_accs)
plt.plot(dev_accs[0], label="Test benign acc")
plt.plot(dev_accs[1], label="Test attack success rate")
plt.plot(dev_accs[2], label="Test attack success rate (replacement only)")
plt.legend()
plt.text(0, 0, "total epochs: {}".format(len(dev_accs[0])))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig(pic_path)