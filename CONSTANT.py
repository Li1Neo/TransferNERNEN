from data_process import CDRProcessor, NCBIProcessor
from NERNEN import BertSoftmaxForNer
from transformers import BertConfig, BertTokenizer
MODEL_CLASSES = {
    # bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSoftmaxForNer, BertTokenizer),
}

DATASET = {
    "ncbi": {
        "from": {
            "train": "NCBI/NCBItrainset_corpus.txt",
            "dev": "NCBI/NCBIdevelopset_corpus.txt",
            "test": "NCBI/NCBItestset_corpus.txt",
        },
        "to": {
            "train": "NCBI/train.txt",
            "dev": "NCBI/dev.txt",
            "test": "NCBI/test.txt",
            "zs_test": "NCBI/zs_test.txt",
        },
    },
    "cdr": {
        "from": {
            "train": "CDR/CDR_TrainingSet.PubTator.txt",
            "dev": "CDR/CDR_DevelopmentSet.PubTator.txt",
            "test": "CDR/CDR_TestSet.PubTator.txt",
        },
        "to": {
            "train": "CDR/train.txt",
            "dev": "CDR/dev.txt",
            "test": "CDR/test.txt",
            "zs_test": "CDR/zs_test.txt",
        },
    },
}

processors = {
    "cdr": CDRProcessor,
    'ncbi': NCBIProcessor
}

