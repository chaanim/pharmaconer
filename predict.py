import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np

from keras_bert import load_trained_model_from_checkpoint, Tokenizer, AdamWarmup, calc_train_steps, get_custom_objects

import keras
from keras import layers, models, optimizers
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

pretrained_path = './data/multi_cased_L-12_H-768_A-12'
#train_path = './data/PharmaCoNER-train-1.1.ascii.notriplebreaks.nersuite'

# config_path = os.path.join(pretrained_path, 'bert_config.json')
# checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

model_path = sys.argv[1]
nersuite_path = sys.argv[2]
out_path = sys.argv[3]

use_crf = True
max_sequence_len = 128

token_dict = {}
with open(vocab_path, 'r') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_inv = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict, cased=True)

import pickle

tag_dict = pickle.load(open(model_path+'.tags.pkl', 'rb'))

inv_tag_dict = {ind: tag for tag, ind in tag_dict.items()}

def grouper(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

# FIXME: DRY, use the function from train.py (move to utils.py or so)
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


def to_conll(input_file, output_file, predictions):
    '''
    THIS IS 100% AD HOC!
    '''
    f = open(input_file, 'r')
    outputs = []
    
    prediction_index = [0,1]
    
    for line in f:
        if line.strip() == '':
            outputs.append('\n')
            prediction_index[0] += 1
            prediction_index[1] = 1 # New sentence starts, index 0 == [CLS] token
            continue
        
        data = line.strip().split('\t')
        assert len(data) >= 3
        token = line.strip().split()[-1]
        tokens = tokenizer.tokenize(token) # Retokenize with sub-word units
        tokens.remove('[CLS]')
        tokens.remove('[SEP]')

#        print(prediction_index)
        tag_index = predictions[prediction_index[0], prediction_index[1]] # Take the tag of the first sub-word unit
        data.append(inv_tag_dict[tag_index])
        outputs.append('\t'.join(data)+'\n')

        prediction_index[1] += len(tokens)
        
        if prediction_index[1] >= max_sequence_len: # If we reach the end of the first sentence chunk, move to the next prediction line
            prediction_index[0] += 1
            # The same original token might continue in the next prediction line,
            # Jump the the beginning of the next token
            # Note: there is no [CLS] token at the beginning of a continued sentence.
            prediction_index[1] = prediction_index[1] - max_sequence_len

    out_f = open(output_file, 'wt')
    out_f.writelines(outputs)
#    import pdb; pdb.set_trace()

print('Loading model')
cu = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
cu.update(get_custom_objects())
ner_model = models.load_model(model_path, custom_objects=cu)

print('Loading data')

sentence_indices, segments, _, _, sentences = generate_data(nersuite_path)

print('Making predictions')

preds = np.argmax(ner_model.predict([sentence_indices, segments], verbose=1), axis=-1)
# pred_tags = [[inv_tag_dict[t] for t in s[:len(sentences[si])]] for si, s in enumerate(preds)] # Remove padding and convert to tags
# flat_pred_tags = [t for sent in pred_tags for t in sent]

to_conll(nersuite_path, out_path, preds)