# %% 
import torch
from generate_anaphora_corpuses import names, female_names, male_names

# %%
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, vocab):
        super().__init__()        
        self.corpus = corpus
        self.vocab = vocab

    def __len__(self):        
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.ids_from_chars(self.corpus[idx])
    
    def ids_from_chars(self, example):
        return torch.tensor([self.vocab.index(word) for word in example.split()])

    def chars_from_ids(self, tensor):
        return " ".join([self.vocab[idx] for idx in tensor])

# %%
def get_word_set(list_of_strings):
    word_set = set()
    for string in list_of_strings:
        words = string.split()
        word_set.update(words)
    return word_set

def generate_anaphora_datasets(path, excluded_females=0, exclude_men=False):
    included_names = female_names[excluded_females:]
    excluded_names = female_names[:excluded_females]

    if exclude_men:
        excluded_names += male_names
    else:
        included_names += male_names

    with open(f"{path}/base.txt", "r") as file:
        corpus_train = file.readlines()
    corpus_test = []

    for name in included_names:
        with open(f"{path}/{name}.txt", "r") as file:
            corpus_train += file.readlines()

    for name in excluded_names:
        with open(f"{path}/{name}.txt", "r") as file:
            corpus_test += file.readlines()

    print(f"Length of corpus train: {len(corpus_train)}")

    vocab = list(get_word_set(corpus_test + corpus_train))
    vocab.sort()

    ds_train  = TextDataset(corpus_train, vocab)
    ds_test = TextDataset(corpus_test, vocab)

    return ds_train, ds_test

def generate_sorting_dataset():
    with open("./sorting/train.txt", "r") as file:
        corpus_train = file.readlines()
    with open("./sorting/test.txt", "r") as file:
        corpus_test = file.readlines()

    print(f"Length of corpus train: {len(corpus_train)}")

    vocab = list(get_word_set(corpus_test + corpus_train))

    ds_train  = TextDataset(corpus_train, vocab)
    ds_test = TextDataset(corpus_test, vocab)

    return ds_train, ds_test