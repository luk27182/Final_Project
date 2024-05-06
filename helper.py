# %%
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt

import einops

# %%
def train_model(model, ds_train, ds_test, num_epochs, print_every=5, batch_size=32):
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=False, padding_value=3))
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=False, padding_value=3))

    assert ds_train.vocab == ds_test.vocab
    vocab = ds_train.vocab


    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    loss_history = []
    test_accuracy_history = []
    for epoch in range(num_epochs):
        avg_loss = 0
        for batch in dl_train:
            loss = 0
            optimizer.zero_grad()
            model_out = model(batch) # Model_Out shape is SEQ X BATCH X VOCAB
            batch = einops.rearrange(batch, 'S B -> B S')
            predictions = 0
            for i, example in enumerate(batch):
                break_point = example.tolist().index(vocab.index("<UNK>"))+1
                loss += criterion(model_out[break_point-1:-1, i], example[break_point:])
                predictions += example[break_point:].size(0)
            loss /= predictions
            avg_loss += loss

            loss.backward()
            optimizer.step()
        avg_loss /= len(dl_train)

        # Calculate the test accuracy
        total = 0
        correct = 0
        for batch in dl_test:
            total += batch.size(1)

            loss = 0
            batch = batch
            model_out = model(batch)
            batch = einops.rearrange(batch, 'S B -> B S')
            model_out = model_out.topk(1)[1].squeeze(-1)
            for i, example in enumerate(batch):
                break_point = example.tolist().index(vocab.index("<UNK>"))+1
                
            for index in range(model_out.size(1)):
                break_point = batch[index].tolist().index(vocab.index("<UNK>"))+1
                if torch.all(model_out[break_point-1:-1, index].eq(batch[index, break_point:])):
                    correct += 1
        test_accuracy = correct / total

        if (epoch+1) % print_every == 0:    
            print(f"Epoch {epoch+1} Train Loss {avg_loss:.5f} Test Accuracy {test_accuracy:.3f}")

        loss_history.append(avg_loss)
        test_accuracy_history.append(test_accuracy)
    return (loss_history, test_accuracy_history)


# %%
def test_example(model, example, ds, max_length=10, verbose=True, be_safe=False):
    device = next(model.parameters()).device
    model.eval()

    if verbose: print(f"===== TESTING NEW EXAMPLE =====")
    
    if be_safe:
        break_point = example.tolist().index(ds.vocab.index("<UNK>"))+1

        input = example[:break_point]
        output = example[break_point:]

        input = input.view(-1, 1)
        while len(input) < max_length and input[-1][0] != ds.vocab.index("<EOS>"):
            model_out  = model(input)
            _, new_out = model_out[:, 0][-1].topk(1)
            input = torch.cat([input, torch.tensor([new_out]).to(device).view(1, 1)])
        
        if verbose: print(f"model out: {ds.chars_from_ids(input.flatten()[break_point:])}")
    
        return ds.chars_from_ids(output) == ds.chars_from_ids(input.flatten()[break_point:])
    else:
        break_point = example.tolist().index(ds.vocab.index("<UNK>"))
        model_out = model(example[:-1].view(-1, 1))
        _, pred = model_out[:, 0].topk(1)
        target = example[break_point+1:]
        if verbose:
            print(f"input: {ds.chars_from_ids(example[:-1].view(-1, 1))}")
            print(f"model out: {ds.chars_from_ids(pred.flatten()[break_point:])}")

        return torch.all(pred.flatten()[break_point:] == target.flatten()).item()
    
# %%
def plot_attn_patterns(model, example, ds):
    test_example(model, example, ds, verbose=False)
    example = example[:-1]

    pattern_keys = [key for key in model.cache.keys() if 'attn' in key]
    for key in pattern_keys:
        attn_patterns = model.cache[key][0]
        num_heads = attn_patterns.size(0)

        fig, ax = plt.subplots(1, num_heads+1, figsize=(20, 5))

        ax[0].imshow(torch.mean(attn_patterns, dim=0), cmap="hot")
        ax[0].set_title(f"{key}AVG")

        ax[0].set_xticks(range(len(example)), ds.chars_from_ids(example).split(" "), rotation=90)
        ax[0].set_yticks(range(len(example)), ds.chars_from_ids(example).split(" "))

        for i in range(num_heads):
            ax[i+1].imshow(attn_patterns[i], cmap="hot")
            ax[i+1].set_title(f"{key}H{i}")

            ax[i+1].set_xticks(range(len(example)), ds.chars_from_ids(example).split(" "), rotation=90)
            ax[i+1].set_yticks(range(len(example)), ds.chars_from_ids(example).split(" "))

        plt.show()

# %% 
def plot_final_patterns(models, examples, ds, loc):
    fig, ax = plt.subplots(1, len(models), figsize=(15, 5))
    for i, (model, example) in enumerate(zip(models, examples)):
        test_example(model, example, ds, verbose=False)
        example = example[:-1]

        if loc == "final":
            final_key = [key for key in model.cache.keys() if 'L1' in key or "symbol_to_text" in key][0]
        else:
            final_key = [key for key in model.cache.keys() if 'L0' in key or "text_to_symbol" in key][0]

        attn_patterns = model.cache[final_key][0]

        ax[i].imshow(torch.mean(attn_patterns, dim=0), cmap="hot")
        ax[i].set_title(f"{final_key}AVG")

        ax[i].set_xticks(range(len(example)), ds.chars_from_ids(example).split(" "), rotation=90)
        ax[i].set_yticks(range(len(example)), ds.chars_from_ids(example).split(" "))

    plt.show()