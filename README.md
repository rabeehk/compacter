# COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers
This repo contains the pytorch implementation of our draft [COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers](https://www.idiap.ch/~rkarimi/papers/compact_adapters.pdf).
This repo additionally contains the implementation for the recent parameter-efficient finetuning methods as well.

# Installation
```
python setup.py install
```

# How to run the models 
We provide the example scripts to run each model in the paper in `seq2seq/scripts`
folder with their config files in `seq2seq/configs`. To run the models, please do
`cd seq2seq` and run:
 - Full-finetuning (T5):
   ```
   bash scripts/baseline.sh
   ```  
 - AdapterDrop: 
   ```
   bash scripts/adapters_drop.sh
   ``` 
 - Adapters:
   ```
   bash scripts/adapters.sh
   ```
 - Low-rank adapters (uses rank-1 approximation for each adapter weight):
   ```
   bash scripts/low_rank_adapters.sh
   ```  
 - Pfeiffer-Adapters:
   ```
   bash scripts/pfeiffer_adapters.sh
   ``` 
 - BitFit:
   ```
   bash scripts/bitfit.sh
   ```
 - Compacter++:
   ```
   bash scripts/compacter++.sh
   ``` 
 - Compacter:
   ```
   bash scripts/compacter.sh
   ```
 - PHM-Adapters:
   ```
   bash scripts/phm_adapters.sh
   ```
 - Intrinsic-SAID:
   ```
   bash scripts/intrinsic_said.sh
   ```
 - Prompt tuning-R (Prompt tuning with random initialization): 
   ```
   bash scripts/prompt_tuning_random_init.sh
   ```
 - Prompt tuning-T (Prompt tuning with initialization from language model's vocabulary):
   ```
   bash scripts/prompt_tuning_tokens_init.sh
   ``` 

## Bibliography
If you find this repo useful, please cite our work:

```
@inproceedings{karimi2021parameterefficient,
  title={Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks},
  author={Karimi Mahabadi, Rabeeh and Ruder, Sebastian and Dehghani, Mostafa and Henderson, James},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```

To implement the intrinsic-SAID method, we used the codes from the following paper. If using this 
method, please also consider citing this work:
```
@inproceedings{aghajanyan-etal-2021-intrinsic,
    title = "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning",
    author = "Aghajanyan, Armen  and
      Gupta, Sonal  and
      Zettlemoyer, Luke",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
To implement parameterized hypercomplex layers, we use the implementation of the following work,
if using PHM-adapters/Compacter/Compacter++ please also consider citing this work:
```
@article{le2021parameterized,
  title={Parameterized hypercomplex graph neural networks for graph classification},
  author={Le, Tuan and Bertolini, Marco and No{\'e}, Frank and Clevert, Djork-Arn{\'e}},
  journal={arXiv preprint arXiv:2103.16584},
  year={2021}
}
```

## Final words
Hope this repo is useful for your research. For any questions, please create an issue or
email rabeeh.k68@gmail.com or rkarimi@idiap.ch, and I will get back to you as soon as possible.

