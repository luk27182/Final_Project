# %%
import numpy as np
np.random.seed(42)

num_used_letters = 26
length_to_sort = 5

train_size = 50000
test_size = 1000


# %%
letters = [l for l in "qwertyuiopasdfghjklzxcvbnm"]
letters.sort()
letters = letters[:num_used_letters]

# %%
with open(f"./sorting/train.txt", "w") as file:
    for example in range(train_size):
        unsorted = np.random.permutation(letters)[:length_to_sort]
        sorted = unsorted.copy()
        sorted.sort()

        example = f"<SOS> {' '.join(unsorted)} <UNK> {' '.join(sorted)} <EOS>"
        file.write(example + "\n")

with open(f"./sorting/test.txt", "w") as file:
    for example in range(test_size):
        unsorted = np.random.permutation(letters)[:length_to_sort]
        sorted = unsorted.copy()
        sorted.sort()

        example = f"<SOS> {' '.join(unsorted)} <UNK> {' '.join(sorted)} <EOS>"
        file.write(example + "\n")

# %%
