import re
import string

import stanza
import spacy

import nltk.corpus
import itertools


STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)

nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)
nlp_spacy = spacy.load("en_core_web_sm")

# schema linking, similar to IRNet
def compute_schema_linking(question, column, table):
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = "CEM"
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = "TEM"

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = "CPM"
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = "TPM"
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}
    qv_match = {}
    qv_match_word = []
    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue

        num_flag = isnumber(word)

        CELL_MATCH_FLAG = "CELLMATCH"
        VALUE_MATCH_FLAG = "VALUEMATCH"
        val_id = 0
        for col_id, column in enumerate(schema.columns):
            if col_id == 0:
                assert column.orig_name == "*"
                continue

            # word is number 
            if num_flag:
                if column.type in ["number", "time"]:  # TODO fine-grained date
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG
                    qv_match[f"{q_id},{val_id}"] = VALUE_MATCH_FLAG
                    val_id += 1
                    qv_match_word.append(word)

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match, "value_match": qv_match, "value_word": qv_match_word}
    return cv_link

def compute_dependency_linking(tokens, type='stanza'):
    dp_linkage = {}
    tokens_joined = ' '.join(tokens)

    if type == 'stanza':
        doc_stanza = nlp_stanza(tokens_joined)
        words_stanza = doc_stanza.sentences[0].words
        for i in range(len(words_stanza)):
            words_stanza[i].id -= 1 
            words_stanza[i].head -= 1

        for i, j in itertools.product(range(len(words_stanza)), repeat=2):
            if i == j:
                continue
            
            # if head's index is 0 > ROOT Node
            # ROOT Node는 
            elif words_stanza[i].head == -1:
                continue

            elif words_stanza[i].head == j:
                dp_linkage[f"{j},{i}"] = 'F'
                dp_linkage[f"{i},{j}"] = 'B'

        dp_link = {"dp_link": dp_linkage}
        
    elif type == 'spacy':
        doc_spacy = nlp_spacy(tokens_joined)
        words_spacy = []

        for idx, token in enumerate(doc_spacy):
            words_spacy.append({"id": idx,
                                "text": token.text,
                                "dep": token.dep_,
                                "head": token.head.i})

        for i, j in itertools.product(range(len(words_spacy)), repeat=2):
            if i == j:
                continue
            
            # if head's index is 0 > ROOT Node
            elif words_spacy[i]['head'] == words_spacy[i]['id']:
                continue

            elif words_spacy[i]['head'] == j:
                dp_linkage[f"{j},{i}"] = 'F'
                dp_linkage[f"{i},{j}"] = 'B'

        dp_link = {"dp_link": dp_linkage}

    return dp_link