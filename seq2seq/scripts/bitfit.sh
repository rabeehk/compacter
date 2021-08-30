# This scripts trains bitfit method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3.
python run_seq2seq.py  configs/bitfit.json
