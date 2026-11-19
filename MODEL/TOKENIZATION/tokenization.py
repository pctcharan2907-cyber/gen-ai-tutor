import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 1. File paths
TRAIN_FILE = "MODEL/DATA/train.jsonl"

OUTPUT_DIR = Path("MODEL/TOKENIZATION/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Special tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

# 3. Read training stories
def read_stories():
    with open(TRAIN_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                story = json.loads(line)["text"]
                if story.strip():
                    yield story

# 4. Create BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 5. Configure pre-tokenizer and decoder
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# 6. Configure BPE trainer
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# 7. Train tokenizer
tokenizer.train_from_iterator(read_stories(), trainer)

# 8. Save complete tokenizer
tokenizer_file = OUTPUT_DIR / "bpe_tokenizer.json"
tokenizer.save(str(tokenizer_file))

# 9. Save vocabulary
vocabulary = tokenizer.get_vocab()

vocab_file = OUTPUT_DIR / "vocabulary.json"

with open(vocab_file, "w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False, indent=2)

# 10. Print vocabulary size
print("Vocabulary size:", len(vocabulary))

# 11. Test tokenization
sentence = "The student is learning artificial intelligence."

encoding = tokenizer.encode(sentence)

print("\nTokens:", encoding.tokens)
print("Token IDs:", encoding.ids)
print("Decoded text:", tokenizer.decode(encoding.ids))

# 12. Confirm files saved
print("\nTokenizer training and saving completed.")
print("Tokenizer saved at:", tokenizer_file.resolve())
print("Vocabulary saved at:", vocab_file.resolve())