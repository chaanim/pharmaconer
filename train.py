import os
import sys
import numpy as np

from keras_bert import load_trained_model_from_checkpoint, Tokenizer, AdamWarmup, calc_train_steps, build_model_from_config

import keras
from keras import layers, models, optimizers
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# pretrained_path = './data/uncased_L-12_H-768_A-12/'
pretrained_path = './data/multi_cased_L-12_H-768_A-12'
#pretrained_path = './data/multilingual_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

use_crf = True
max_sequence_len = 128
batch_size = 16
weight_decay = float(sys.argv[2])
print('Weight decay: %s' % weight_decay)
#save_path = './models/ner_model.h5'
save_path = sys.argv[1]

# train_path = './data/eng.train.flipped'
# devel_path = './data/eng.testa.flipped'
#train_path = './data/PharmaCoNER-train-1.1.ascii.notriplebreaks.nersuite'
#devel_path = './data/PharmaCoNER-dev-1.1.ascii.notriplebreaks.nersuite'
train_path = './data/PharmaCoNER-train-1.1.nersuite'
devel_path = './data/PharmaCoNER-dev-1.1.nersuite'
#os.environ['TF_KERAS'] = '1'

class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, name, data, model_path=None):
        super().__init__()
        
        self.name = name
        self.data = data # Data should be (sentence_indices, segments, tags, sentences)
        self.model_path = model_path
        self.best_f = 0.0

    def on_epoch_end(self, epoch, logs={}):
        print('=== Evaluating %s, epoch: %s ===' % (self.name, epoch))
        
        preds = np.argmax(self.model.predict([self.data[0], self.data[1]]), axis=-1)
        pred_tags = [[inv_tag_dict[t] for t in s[:len(self.data[-1][si])]] for si, s in enumerate(preds)]
        flat_pred_tags = [t for sent in pred_tags for t in sent]
        
        gold_tags = [[inv_tag_dict.get(t[0], 'O') for t in s[:len(self.data[-1][si])]] for si, s in enumerate(self.data[2])]
        flat_gold_tags = [t for sent in gold_tags for t in sent]
        
        from conlleval import evaluate
        f_score = evaluate(flat_gold_tags, flat_pred_tags, verbose=True)[-1]
        if f_score > self.best_f:
            self.best_f = f_score
            print('=== Best %s F-score so far ===' % self.name)
            if self.model_path:
                self.model.save(save_path)



token_dict = {}
with open(vocab_path, 'r') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_inv = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict, cased=True) # cased = True will not canonicalize unicode characters

def read_tags(path):
    f = open(path, 'r')
    tags = set(l.split()[0] for l in f if l.strip() != '')
    return {tag: index for index, tag in enumerate(tags)}
# def read_tags(path):
#     """
#     Reads all NER tags from the training data.
#     """
#     ann_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ann')]
#     
#     tags = set()
#     for f in ann_files:
#         for line in open(f, 'r'):
#             if line.startswith('#') or line.strip() == '':
#                 continue
#             tags.add('B-'+line.strip().split()[1])
#             tags.add('I-'+line.strip().split()[1])
# 
#     tags.add('O')
#         
#     tag_dict = {tag: index for index, tag in enumerate(tags)}
#     return tag_dict

tag_dict = read_tags(train_path)

import pickle

pickle.dump(tag_dict, open(save_path+'.tags.pkl', 'wb'))

inv_tag_dict = {ind: tag for tag, ind in tag_dict.items()}

def grouper(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def generate_data(path, has_tags=True):
    '''
    Read brat format data and convert to Keras format.
    Tiny dataset, no need for an actual generator.
    
    THIS IS 100% AD HOC!
    '''
    f = open(path, 'r')
    sentences, tags, token_alignment = [], [], []
    
    sentence_tokens, sentence_tags, sentence_token_alignment = ['[CLS]'], ['O'], []
    
    token_index = 0
    # This is used for aligning the tokenizations
    # TODO: Add offsets to keep track of the retokenized alignment
    for line in f:
        if line.strip() == '':
            sentence_tokens.append('[SEP]')
            sentence_tags.append('O')
            
            # Simply split too long sentences into chunks, we don't add new [CLS] or [SEP] tokens. Could also split inside an entity...
            token_chunks = grouper(sentence_tokens, max_sequence_len)
            tag_chunks = grouper(sentence_tags, max_sequence_len)
            for token_chunk, tag_chunk in zip(token_chunks, tag_chunks):
                sentences.append(token_chunk)
                tags.append(tag_chunk)

            sentence_tokens, sentence_tags = ['[CLS]'], ['O']
            continue
        #tag, beg, end, token = line.strip().split('\t')
        if has_tags:
            tag = line.strip().split()[0]
        else: # If we are reading e.g. test data, the tag column is missing
            tag = 'O'
        token = line.strip().split()[-1]
        # FIXME: Keras-BERT tokenizer converts unicode characters to canonical forms, e.g. ä -> a
        # Multilingual BERT model however has the original characters in the vocab.
        tokens = tokenizer.tokenize(token) # Retokenize with sub-word units
        tokens.remove('[CLS]')
        tokens.remove('[SEP]')
        for i, t in enumerate(tokens):
            if not t in token_dict:
                print('UNK TOKEN %s' % t)
                t = '[UNK]'
            sentence_tokens.append(t)
            if i != 0 and tag.startswith('B-'):
                sentence_tags.append(tag.replace('B-', 'I-')) # Only the first new token preserves B tag
            else:
                sentence_tags.append(tag)
    # import pdb; pdb.set_trace()
    sentence_indices = [[token_dict[token] for token in tokens] + [0] * (max_sequence_len - len(tokens)) for tokens in sentences]
    token_weights = np.array([[1] * len(tokens) + [0] * (max_sequence_len - len(tokens)) for tokens in sentences])
    segments = [[0]*max_sequence_len for tokens in sentences] # We don't care about the segments

    if use_crf:
        tag_indices = np.expand_dims(np.array([[tag_dict[t] for t in tokens] + [-1] * (max_sequence_len - len(tokens)) for tokens in tags]), -1)
    else:
        tag_indices = np.expand_dims(np.array([[tag_dict[t] for t in tokens] + [tag_dict['O']] * (max_sequence_len - len(tokens)) for tokens in tags]), -1)
    return sentence_indices, segments, tag_indices, token_weights, sentences

train_sentence_indices, train_segments, train_tags, train_token_weights, train_sentences = generate_data(train_path)
devel_sentence_indices, devel_segments, devel_tags, devel_token_weights, devel_sentences = generate_data(devel_path)

# print("Baseline CRF model")
# 
# inp = layers.Input(shape=(max_sequence_len, ))
# emb = layers.Embedding(len(token_dict), 50, mask_zero=True)(inp)
# crf = CRF(len(tag_dict), sparse_target=True)(emb)
# base_model = models.Model(inputs=inp, outputs=crf)
# base_model.compile(optimizers.Adam(lr=0.01), crf_loss, metrics=[crf_viterbi_accuracy])
# 
# base_model.summary()
# 
# base_model.fit([train_sentence_indices], train_tags, validation_data=([devel_sentence_indices], devel_tags), batch_size=batch_size, epochs=50, verbose=1)


print("Loading BERT")

total_steps, warmup_steps = calc_train_steps(
    num_example=len(train_sentences),
    batch_size=batch_size,
    epochs=10,
    warmup_proportion=0.1,
)

print(total_steps, warmup_steps)

optimizer = AdamWarmup(5*total_steps, warmup_steps, lr=2e-5, min_lr=2e-7, weight_decay=weight_decay)

# import pdb; pdb.set_trace()
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False, trainable=True, seq_len=max_sequence_len)
# bert_model, _ = build_model_from_config(config_path, training=False, trainable=True, seq_len=max_sequence_len)
#bert_model.summary(line_length=120)

if use_crf:
    #prediction_layer = layers.Dense(768, activation='tanh')(bert_model.output)
    prediction_layer = CRF(len(tag_dict), sparse_target=True)(bert_model.output)

    ner_model = models.Model(inputs=bert_model.inputs, outputs=prediction_layer)
    ner_model.compile(optimizer, crf_loss, metrics=[crf_viterbi_accuracy]) # optimizers.Adam(lr=2e-5)
else:
    # prediction_layer = layers.Dense(768, activation='tanh')(bert_model.output)
    prediction_layer = layers.Dense(len(tag_dict), activation='softmax')(bert_model.output)

    ner_model = models.Model(inputs=bert_model.inputs, outputs=prediction_layer)
    ner_model.compile(optimizer, 'sparse_categorical_crossentropy', sample_weight_mode='temporal')

ner_model.summary(line_length=120)

# import pdb; pdb.set_trace()
train_cb = EvaluationCallback('TRAIN', (train_sentence_indices, train_segments, train_tags, train_sentences))
devel_cb = EvaluationCallback('DEVEL', (devel_sentence_indices, devel_segments, devel_tags, devel_sentences), save_path)

print("Training")
if use_crf:
    ner_model.fit([train_sentence_indices, train_segments], train_tags, validation_data=([devel_sentence_indices, devel_segments], devel_tags), batch_size=batch_size, epochs=50, verbose=2, callbacks=[train_cb, devel_cb])
else:
    ner_model.fit([train_sentence_indices, train_segments], train_tags, sample_weight=train_token_weights, validation_data=([devel_sentence_indices, devel_segments], devel_tags, devel_token_weights), batch_size=batch_size, epochs=10, verbose=1, callbacks=[train_cb, devel_cb])
#    ner_model.fit([train_sentence_indices[:100], train_segments[:100]], train_tags[:100], sample_weight=train_token_weights[:100], validation_data=([devel_sentence_indices[:100], devel_segments[:100]], devel_tags[:100], devel_token_weights[:100]), batch_size=batch_size, epochs=1, verbose=1)

# import pdb; pdb.set_trace()
