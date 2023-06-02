from typing import Type

from aprec.recommenders.sequential.models.generative.tokenizers.id_tokenizer import IDTokenizer
from ..tokenizer import Tokenizer
from .svd_tokenizer import SVDTokenizer

def get_tokenizer_class(classname) -> Type[Tokenizer]:
    if classname == "svd":
        return SVDTokenizer
    elif classname == "id":
        return IDTokenizer