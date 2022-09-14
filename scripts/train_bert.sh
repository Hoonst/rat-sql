export CUDA_VISIBLE_DEVICES=${1:-4}
python run.py train experiments/spider-bert-run.jsonnet