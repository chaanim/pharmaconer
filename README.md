# Named entity recognition with BERT

Tools for finetuning BERT for named entity recognition, originally tested on PharmacoNER data.

Requirements:

1. a pretrained BERT model (model paths hardcoded in the code at the moment)
2. conlleval.py file from https://github.com/sighsmile/conlleval
3. Train and devel files in the format below (paths hardcoded currently)

Data should be in conllish format (1st column tag, last column token (will get retokenized to subword units)):

~~~~
I-MISC West
I-MISC Indian
O all-rounder
I-PER Phil
I-PER Simmons
O took
O four
O for
O 38
O on
O Friday
O as
I-ORG Leicestershire
O beat
I-ORG Somerset
O by
O an
O innings
O and
O 39
O runs
O in
O two
O days
O to
O take
O over
O at
O the
O head
O of
O the
O county
O championship
O .	
~~~~

train.py trains a new model, requires model save path and weight decay as input, e.g.
~~~~
python3 train.py ./models/model.h5 0.01 | tee ./logs/model.log
~~~~

predict.py makes predictions with an existing model, requires model path, input path and output path, e.g.
~~~~
python3 predict.py ./models/model.h5 ./data/PharmaCoNER-dev-1.1.nersuite ./predictions.nersuite
~~~~

The predicted tags are inserted to the last column of the input file.