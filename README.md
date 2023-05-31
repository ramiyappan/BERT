# Implement your Own BERT
This is an exercise in developing a minimalist version of BERT.

In this assignment, you will implement some important components of the BERT model to gain a better understanding its architecture. 
You will then perform sentence classification on two datasets: the ``sst`` dataset and the ``cfimdb`` dataset with your BERT model.

## Assignment Details

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies. Make sure to do the rest of your work on the appropriate environment.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use `torch`, no other external libraries are allowed (e.g., `transformers`).
* We will run your code with the following commands, so make sure that whatever your best results are reproducible using these commands (where you replace GMUID with your andrew ID):
```
mkdir -p GMUID

python3 classifier.py --option [pretrain/finetune] --epochs NUM_EPOCHS --lr LR --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt
```

## Reference accuracies: 

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

Pretraining for SST:
Dev Accuracy: 0.391 (0.007)
Test Accuracy: 0.403 (0.008)

Finetuning for SST:
Dev Accuracy: 0.515 (0.004)
Test Accuracy: 0.526 (0.008)

Finetuning for CFIMDB:
Dev Accuracy: 0.966 (0.007)
Test Accuracy: -

### Acknowledgements
_This assignment is adapted from the Carnegie Mellon University's CS11-711 course and the minBERT assignment created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt and Brendon Boldt._

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

