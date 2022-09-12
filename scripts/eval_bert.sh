export CUDA_VISIBLE_DEVICES=${1:-4}
python run.py eval experiments/spider-bert-run.jsonnet