import argparse
import json

import _jsonnet
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql.utils import registry
# noinspection PyUnresolvedReferences
from ratsql.utils import vocab


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        # enc_dec.py/ class EncDecModel
        # EncDecModel 내 Preproc을 따름
        # EncDecModel 내 Preproc은 Encoder / Decoder 각기의 Preproc을 모두 사용
        # 즉, Preproc들을 통해 Encoder에 필요한 데이터, Decoder에 필요한 데이터를 각기 전처리 수행

    def preprocess(self):
        self.model_preproc.clear_items()
        # def clear_items(self):
        #    self.texts = collections.defaultdict(list)
        # {'train': [],
        #  'val': []

        for section in self.config['data']:
            # section: train, val


            data = registry.construct('dataset', self.config['data'][section])

            for item in tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True):
                ## 하나의 자연어 질의와 db 정보
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                '''
                def validate_item(self, item, section):
                    return True, None
                '''
                if to_add:
                    # add_item은 preprocess_item이 적용된 item (json line)들을 추가
                    # 'train', 'val' key로 이루어진 Dictionary List에 하나씩 추가
                    # process_item에서 schema_linking과 cell value matching이 진행된다. 
                    # spider_match_utils.py
                    #   > compute_schema_linking
                    #   > compute_cell_value_linking
                    self.model_preproc.add_item(item, section, validation_info)
                    
        self.model_preproc.save()


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()


if __name__ == '__main__':
    args = add_parser()
    main(args)
