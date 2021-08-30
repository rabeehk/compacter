# This scripts trains prompt tuning with initialization from language model's vocabulary method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3.
python run_seq2seq.py  configs/prompt_tuning_tokens_init.json
