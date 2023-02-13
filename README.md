# Implement your Own BERT Assignment
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

### Submission
The submission file should be a zip file with the following structure (assuming your GMU id is ``GMUID``):
```
GMUID/
├── base_bert.py
├── bert.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── setup.py
├── sst-dev-output.pretrain.txt 
├── sst-test-output.pretrain.txt 
├── cfimdb-dev-output.pretrain.txt 
├── cfimdb-test-output.pretrain.txt 
├── sst-dev-output.finetune.txt 
├── sst-test-output.finetune.txt 
├── cfimdb-dev-output.finetune.txt 
└── cfimdb-test-output.finetune.txt 
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It will throw assertion errors if the format is not expected, and we will *not accept submissions that fail this check*. Usage: (1) To create and check a zip file with your outputs, run `python3 prepare_submit.py path/to/your/output/dir GMUID`, (2) To check your zip file, run `python3 prepare_submit.py path/to/your/submit/zip/file.zip GMUID`

### Grading
* A+: You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements. Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the MLM objective to do domain adaptation
    * try [alternative fine-tuning algorithms](https://www.aclweb.org/anthology/2020.acl-main.197)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code that achieves comparable accuracy to our reference implementation
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code but accuracy is not comparable to the reference.
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (bert implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

### Acknowledgements
_This assignment is adapted from the Carnegie Mellon University's CS11-711 course and the minBERT assignment created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt and Brendon Boldt._

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

