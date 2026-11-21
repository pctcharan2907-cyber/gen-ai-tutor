from collections import Counter

# STEP 1: Prepare a small training corpus
corpus = [
    "low",
    "lower",
    "lowest",
    "low",
    "lower",
]

# STEP 2: Count word frequencies
word_freq = Counter(corpus)

# STEP 3: Split each word into characters
word_tokens = {
    word: list(word)
    for word in word_freq
}

print("INITIAL WORD REPRESENTATION")
print("---------------------------")
for word, tokens in word_tokens.items():
    print(word, ":", tokens, "Frequency:", word_freq[word])


# STEP 4: Count adjacent token pairs
def get_pair_frequencies(word_tokens, word_freq):
    pair_counts = Counter()

    for word, tokens in word_tokens.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += word_freq[word]

    return pair_counts


# STEP 5: Merge a selected pair throughout the corpus
def merge_pair(word_tokens, best_pair):
    new_word_tokens = {}

    for word, tokens in word_tokens.items():
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        new_word_tokens[word] = merged_tokens

    return new_word_tokens


# STEP 6: Repeat the BPE merging process
number_of_merges = 5
learned_merges = []

print("\nBPE MERGING PROCESS")
print("-------------------")

for step in range(number_of_merges):
    pair_counts = get_pair_frequencies(word_tokens, word_freq)

    if not pair_counts:
        break

    best_pair = max(pair_counts, key=pair_counts.get)
    frequency = pair_counts[best_pair]

    print(f"\nMerge {step + 1}")
    print("Most frequent pair:", best_pair)
    print("Frequency:", frequency)

    word_tokens = merge_pair(word_tokens, best_pair)
    learned_merges.append(best_pair)

    print("Updated representation:")
    for word, tokens in word_tokens.items():
        print(word, ":", tokens)


# STEP 7: Display the final vocabulary
vocabulary = set()

for tokens in word_tokens.values():
    vocabulary.update(tokens)

print("\nFINAL BPE VOCABULARY")
print("--------------------")
print(sorted(vocabulary))

print("\nLearned merge rules:")
for merge in learned_merges:
    print(merge[0], "+", merge[1], "->", merge[0] + merge[1])