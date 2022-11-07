# RAT-SQL

This repository contains code for the ACL 2020 paper ["RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers"](https://arxiv.org/abs/1911.04942).

If you use RAT-SQL in your work, please cite it as follows:
``` bibtex
@inproceedings{rat-sql,
    title = "{RAT-SQL}: Relation-Aware Schema Encoding and Linking for Text-to-{SQL} Parsers",
    author = "Wang, Bailin and Shin, Richard and Liu, Xiaodong and Polozov, Oleksandr and Richardson, Matthew",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "7567--7578"
}
```

## Changelog

**2020-08-14:**
- The Docker image now inherits from a CUDA-enabled base image.
- Clarified memory and dataset requirements on the image.
- Fixed the issue where token IDs were not converted to word-piece IDs for BERT value linking.  

## Usage

### Step 1: Download third-party datasets & dependencies

Download the datasets: [Spider](https://yale-lily.github.io/spider) and [WikiSQL](https://github.com/salesforce/WikiSQL). In case of Spider, make sure to download the `08/03/2020` version or newer.
Unpack the datasets somewhere outside this project to create the following directory structure:
```
/path/to/data
├── spider
│   ├── database
│   │   └── ...
│   ├── dev.json
│   ├── dev_gold.sql
│   ├── tables.json
│   ├── train_gold.sql
│   ├── train_others.json
│   └── train_spider.json
└── wikisql
    ├── dev.db
    ├── dev.jsonl
    ├── dev.tables.jsonl
    ├── test.db
    ├── test.jsonl
    ├── test.tables.jsonl
    ├── train.db
    ├── train.jsonl
    └── train.tables.jsonl
```

To work with the WikiSQL dataset, clone its evaluation scripts into this project:
``` bash
mkdir -p third_party
git clone https://github.com/salesforce/WikiSQL third_party/wikisql
```

### Step 2: Build and run the Docker image

We have provided a `Dockerfile` that sets up the entire environment for you.
It assumes that you mount the datasets downloaded in Step 1 as a volume `/mnt/data` into a running image.
Thus, the environment setup for RAT-SQL is:
``` bash
docker build -t ratsql .
docker run --rm -m4g -v /path/to/data:/mnt/data -it ratsql
```
Note that the image requires at least 4 GB of RAM to run preprocessing.
By default, [Docker Desktop for Mac](https://hub.docker.com/editions/community/docker-ce-desktop-mac/) and [Docker Desktop for Windows](https://hub.docker.com/editions/community/docker-ce-desktop-windows) run containers with 2 GB of RAM.
The `-m4g` switch overrides it; alternatively, you can increase the default limit in the Docker Desktop settings.

> If you prefer to set up and run the codebase without Docker, follow the steps in `Dockerfile` one by one.
> Note that this repository requires Python 3.7 or higher and a JVM to run [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).

### Step 3: Run the experiments

Every experiment has its own config file in `experiments`.
The pipeline of working with any model version or dataset is: 

``` bash
python run.py preprocess experiment_config_file  # Step 3a: preprocess the data
python run.py train experiment_config_file       # Step 3b: train a model
python run.py eval experiment_config_file        # Step 3b: evaluate the results
```

Use the following experiment config files to reproduce our results:

* Spider, GloVE version: `experiments/spider-glove-run.jsonnet`
* Spider, BERT version (requires a GPU with at least 16GB memory): `experiments/spider-bert-run.jsonnet`
* WikiSQL, GloVE version: `experiments/wikisql-glove-run.jsonnet`

The exact model accuracy may vary by ±2% depending on a random seed. See [paper](https://arxiv.org/abs/1911.04942) for details.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Conducted Experiments (Hoonst)
모든 실험의 절차는 다음과 같음
``` bash
bash train_bert.sh $GPU_ID                       # Train Step
bash eval_bert.sh $GPU_ID                        # Evaluation Step
bash evaluation_format.sh                        # Format Evaluated Results
```

## Train
훈련을 진행할 때는 train_bert.sh를 활용
각 실험들은 파라미터들을 experiments/spider-bert-run.jsonnet 에서 변화를 주어서 진행됨
(변화 시도와 모델 변경을 통해 새로 삽입한 파라미터에 대하여 소개)

### model_config_args
* **bs**: batch size
* **num_layers**: Relation Aware Self Attention Layer 개수
    * Default: 8
    * DT-Fixup를 기반으로 8 -> 24로 향상시켰을 때 어떻게 성능이 변화할 지 측정
* **loss**: Loss Type
    * Default: 'Softmax'
    * Default Softmax를 사용하게 되면 Gradient Explosion이 나타난다는 Github issue가 존재
    * 이를 해고하기 위하여 [근거](https://github.com/microsoft/rat-sql/issues/11)에서 'label_smooth'를 사용하라고 제안
    * Loss 변경 시 안정적 Training 가능
* **plm_version**
    * "bert-large-uncased-whole-word-masking"
    * "google/electra-large-discriminator" - ELECTRA를 활용하기 위해선 plm_lr을 1e-4로 변경 필요 (직접 실험하지는 않았지만, 문헌 과 다른 구현체의 파라미터 참고했을때의 수치)
    * lr을 3e-6으로 설정하게 되면 지속적으로 성능 하락 발생
    * "microsoft/deberta-v3-large"
    * "microsoft/deberta-large"
* **sc_link / cv_link**: schema_linking / column value matching
    * Selection: (True / False)
* **qv_link**
    * Value Matching Experiment
        * question과 column내의 value linking을 수행 > value가 column 내에 포함되어 있으면 입력 Sequence에 더하는 기법
    * Selection: (True / False)
* **dp_link**
    * DP Relation Add Experiment
        * dependency parsing으로 인한 Relation을 추가
    * dp_link를 활성화 하기 위해선 dependency parsing으로 미리 전처리를 수행해놔야 한다.
    * 즉, STANZA / SPACY와 같은 패키지로 dependency parsing을 진행하여 relation을 미리 설정해두어야 한다.
    * Selection: (STANZA / SPACY / False)
* **dist_relation**
    * dist_relation remove experiment
        * 'qq_dist', 'cc_dist', 'tt_dist'와 같이 입력 시퀀스 내 각 토큰들의 거리를 기반으로 만든 Relation을 유지할지에 대한 여부 
    * 유지하지 않으면 모두 default relation으로 치환
    * 필요 실험: 각 dist를 독립적으로 하나씩 빼보는 실험은 진행
    * Selection: (True / False)
* **use_orthogonal**
    * 기존 Loss에 Orthogonal Constraint를 부여
    * Selection: (True / False)
* **use_orth_init**
    * Relation Embedding을 처음부터 Orthogonal하게 설정하는 방법
    * Selection: (True / False)
* **bi_way / bi_match**
    * Relation Control 실험
    * Bi_Way > Uni_Way: Bi-directional한 현재 Relation을 단방향으로 제어
        * ex) qc_default / cq_default를 모두 하나의 default로 변경
    * Bi_Match > Uni_Match: Exact / Partial Match를 모두 하나의 Match로 변경
        * qcCEM , cqCEM > qcCEM 하나의 Relation을 활용하도록 설정
    * Selection: (True / False)
* **att_seq**
    * ANNA [ANNA: Enhanced Language Representation for Question Answering](https://aclanthology.org/2022.repl4nlp-1.13.pdf)에 기반하여 Neighbor-aware Attention을 적용
    * 구현 상의 문제는 없는 것 같으나, 성능이 도출되지 않음
    * Selection: (MP: Multihead Only / MNP: Multihead-Neighbor / NP: Neighbor Only)