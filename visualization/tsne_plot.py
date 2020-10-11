#!/usr/bin/env python

import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

csv.field_size_limit(sys.maxsize)
   
feature_loc = '/home/ubuntu/coco_caption/train.feature.tsv'
label_loc = '/home/ubuntu/coco_caption/train.label.tsv'


def main():
    max_examples = 1000
    points = []
    labels = []
    print("Extracting points")
    current_points = 0
    with open(feature_loc, "r") as tsv_in_file:
        for line in tsv_in_file:
            item = eval(line.split("\t")[-1])
            item['num_boxes'] = int(item['num_boxes'])
            features = np.frombuffer(base64.decodestring(item["features"].encode()), 
                  dtype=np.float32).reshape((item['num_boxes'],-1))
            current_points += 1
            if current_points > max_examples:
                break
            points.extend(features)
            print("{} / {}".format(current_points, max_examples))

    print("Extracting labels")
    current_labels = 0
    with open(label_loc, "r") as tsv_in_file:
        for line in tsv_in_file:
            item = eval(line.split("\t")[-1])
            current_labels += 1
            if current_labels > max_examples:
                break
            labels.extend([x["class"] for x in item])
            print("{} / {}".format(current_labels, max_examples))

    label_set = set(labels)
    colormap = {}
    for idx, unique_label in enumerate(label_set):
        colormap[unique_label] = idx
    label_colors = [colormap[x] for x in labels]
    print(label_colors)

    # cull points
    keep_indices = []
    for i in range(len(label_colors)):
        if label_colors[i] < 30:
            keep_indices.append(i)
    points = [points[x] for x in keep_indices]
    label_colors = [label_colors[x] for x in keep_indices]
    labels = [labels[x] for x in keep_indices]

    # do tsne
    points = np.array(points)
    print(points.shape)
    print(points[0])
    X_embedded = TSNE(n_jobs=-1).fit_transform(points)
    print(X_embedded.shape)

    # form groups
    groups = {}
    for idx, label in enumerate(labels):
        if label not in groups:
            groups[label] = {"X": [], "Y": [], "colors": []}
        groups[label]["X"].append(X_embedded[idx, 0])
        groups[label]["Y"].append(X_embedded[idx, 1])
        groups[label]["colors"].append(colormap[label])
    for label, d in groups.items():
        plt.scatter(d["X"], d["Y"], label=label, marker="o", alpha=0.5)
    plt.legend()
    plt.savefig("plot.png")

if __name__ == "__main__":
        main()
