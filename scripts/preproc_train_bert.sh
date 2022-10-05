export CUDA_VISIBLE_DEVICES=$1
python run.py preprocess experiments/spider-bert-run.jsonnet

python run.py train experiments/spider-bert-run.jsonnet