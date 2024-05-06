# %% Establish basic components
import numpy as np

complex_base_size = 50000
intransitive_verbs = ['runs', 'sleeps', 'laughs', 'cries', 'talks', 'jumps', 'dances', 'sings']
transitive_verbs = ['eats', 'sees', 'hugs', 'paints', 'kicks', 'throws', 'compliments']
female_names = ['Alice', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Abigail', 'Emily', 'Elizabeth', 'Mila']
male_names = ['Bob', 'John', 'Noah', 'Oliver', 'Elijah', 'William', 'James', 'Benjamin', 'Lucas', 'Henry', 'Michael']

names = female_names+male_names
verbs = intransitive_verbs+transitive_verbs



# %% create simple corpus, by subj name
def gen_simple_corpus():
    corpus = []
    for subj in names:
        for verb in intransitive_verbs:
            new_sentence = " ".join([subj, verb])
            parsed = " ".join([verb, subj])
            seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
            corpus.append(seq)

    for subj in names:
        for verb in transitive_verbs:
            for obj in names:
                new_sentence = " ".join([subj, verb, obj])
                parsed = " ".join([verb, subj, obj])
                seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
                corpus.append(seq)
    
    return corpus

def gen_simple_reflexive(subj):
    corpus = []
    for verb in transitive_verbs:
        if subj in female_names:
            new_sentence = " ".join([subj, verb, "herself"])
        else:
            new_sentence = " ".join([subj, verb, "himself"])
        parsed = " ".join([verb, subj, subj])
        seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
        corpus.append(seq)
    return corpus


corpus = gen_simple_corpus()
with open(f"./simple/base.txt", "w") as file:
    for example in corpus:
        file.write(example + "\n")

for name in names:
    corpus = gen_simple_reflexive(name)

    with open(f"./simple/{name}.txt", "w") as file:
        for example in corpus:
            file.write(example + "\n")
# %% generate the complex corpus
type1_phrases = []
for name1 in names:
    for name2 in names:
        type1_phrases.append((f"{name1} friend of {name2}", name1))

type2_phrases = []
for name1 in names:
    for name2 in names:
        type2_phrases.append((f"{name1} thinks that {name2}", name2))

basic_phrases = [(name, name) for name in names]

total_begginings = basic_phrases + type1_phrases + type2_phrases


def gen_complex_corpus():
    complex_corpus = []

    for phrase, subj in total_begginings:
        for verb in intransitive_verbs:
            new_sentence= phrase + " " + verb
            parsed = verb + " " + subj
            seq = "<SOS> " + new_sentence + " <UNK> "+ parsed +" <EOS>"
            complex_corpus.append(seq)
        
    for phrase, subj in total_begginings:
        for verb in transitive_verbs:
            for obj in names:
                new_sentence = f"{phrase} {verb} {obj}"
                parsed = f"{verb} {subj} {obj}"

                seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
                complex_corpus.append(seq)
    complex_corpus = np.random.permutation(complex_corpus)

    return complex_corpus[:complex_base_size] # give a small subset

def gen_complex_reflexive(subj):
    complex_corpus = []
    for phrase, subj in total_begginings:
        if subj != name:
            continue
        for verb in transitive_verbs:
            if subj in female_names:
                new_sentence = phrase + " " + verb + " herself"
            else:
                new_sentence = phrase + " " + verb + " himself"
            parsed = f"{verb} {subj} {subj}"
            seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
            complex_corpus.append(seq)

    return complex_corpus

complex_corpus = gen_complex_corpus()
with open(f"./complex/base.txt", "w") as file:
    for example in complex_corpus:
        file.write(example + "\n")

for name in names:
    complex_corpus = gen_complex_reflexive(name)

    with open(f"./complex/{name}.txt", "w") as file:
        for example in complex_corpus:
            file.write(example + "\n")
# %%
