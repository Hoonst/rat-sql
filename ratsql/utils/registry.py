import collections
import collections.abc
import inspect
import sys


_REGISTRY = collections.defaultdict(dict)

'''
* register
* lookup
* construct
* instantiate
'''

def register(kind, name):
    kind_registry = _REGISTRY[kind]

    def decorator(obj):
        if name in kind_registry:
            raise LookupError(f'{name} already registered as kind {kind}')
        kind_registry[name] = obj
        return obj

    return decorator


def lookup(kind, name):
    if isinstance(name, collections.abc.Mapping):
        # name에 config가 들어가게 되면 instance이므로 
        # name 내에 속해있는 model name을 가져온다.
        '''
        'min_freq': 4,
        'save_path': 'data/spider/nl2code-glove,cv_link=true',
        'word_emb': {'kind': '42B', 'lemmatize': True, 'name': 'glove'}},
        'name': 'EncDec'}
        '''
        name = name['name']

    if kind not in _REGISTRY:
        raise KeyError(f'Nothing registered under "{kind}"')
    return _REGISTRY[kind][name]


def construct(kind, config, unused_keys=(), **kwargs):
    return instantiate(
            lookup(kind, config),
            config,
            unused_keys + ('name',),
            **kwargs)


def instantiate(callable, config, unused_keys=(), **kwargs):
    ## callable: instantiate
    merged = {**config, **kwargs}
    signature = inspect.signature(callable.__init__)

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            raise ValueError(f'Unsupported kind for param {name}: {param.kind}')    

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable(**merged)

    missing = {}
    for key in list(merged.keys()):
        if key not in signature.parameters:
            if key not in unused_keys:
                missing[key] = merged[key]
            merged.pop(key)
    if missing:
        print(f'WARNING {callable}: superfluous {missing}', file=sys.stderr)
    return callable(**merged)
