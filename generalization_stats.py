"""
Checks how many of the devel set entities appear in train and what is the recall for the unseen entities compared to seen.-
Uses .ann file format.
"""

import sys
import os

train_path = sys.argv[1]
devel_path = sys.argv[2]
devel_pred_path = sys.argv[3]

def get_entities(path):
    entities = []
    for f in os.listdir(path):
        if not f.endswith('.ann'):
            continue
        for line in open(os.path.join(path, f)):
            if line.startswith('T'):
                entities.append(tuple([f, ] + line.strip().split('\t')[1:]))
    return set(entities)

train_entities = get_entities(train_path)
devel_entities = get_entities(devel_path)
devel_pred_entities = get_entities(devel_pred_path)

# import pdb; pdb.set_trace()
precision = len(devel_pred_entities.intersection(devel_entities)) / len(devel_pred_entities)
recall = len(devel_pred_entities.intersection(devel_entities)) / len(devel_entities)
f_score= 2 * precision * recall / (precision + recall)

print('precision, recall, f_score')
print(precision, recall, f_score)

unique_train_entities = set(e[-1] for e in train_entities)
unique_devel_entities = set(e[-1] for e in devel_entities)

print('Unique entities train: %s, devel: %s' % (len(unique_train_entities), len(unique_devel_entities)))

print('Unique devel entities also in train: %s' % (len([e for e in unique_devel_entities if e in unique_train_entities])))

seen_devel_entities = set(e for e in devel_entities if e[-1] in unique_train_entities)
unseen_devel_entities = set(e for e in devel_entities if not e[-1] in unique_train_entities)

print('Devel entity occurrences: %s' % len(devel_entities))
print('Seen entity occurrences: %s' % len(seen_devel_entities))
print('Unseen entity occurrences: %s' % len(unseen_devel_entities))

print('Seen recall: %s' % (len(devel_pred_entities.intersection(seen_devel_entities)) / len(seen_devel_entities)))
print('Unseen recall: %s' % (len(devel_pred_entities.intersection(unseen_devel_entities)) / len(unseen_devel_entities)))

# import pdb; pdb.set_trace()