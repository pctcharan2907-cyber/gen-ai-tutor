from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter
import sentencepiece as spm


# Create a multilingual DSA corpus
corpus = {
    "English": [
        "Data structures organize data efficiently",
        "Arrays store elements in contiguous memory",
        "Linked lists consist of connected nodes",
        "Stacks follow the last in first out principle",
        "Queues follow the first in first out principle",
    ],
    "Telugu": [
        "డేటా స్ట్రక్చర్స్ అనేవి డేటాను క్రమబద్ధంగా నిల్వ చేస్తాయి",
        "అర్రే అనేది ఒకే రకమైన డేటాను నిల్వ చేస్తుంది",
        "లింక్డ్ లిస్ట్ నోడ్స్ తో నిర్మించబడుతుంది",
        "స్టాక్ లాస్ట్ ఇన్ ఫస్ట్ అవుట్ సూత్రాన్ని అనుసరిస్తుంది",
        "క్యూ ఫస్ట్ ఇన్ ఫస్ట్ అవుట్ సూత్రాన్ని అనుసరిస్తుంది",
    ],
    "Hindi": [
        "डेटा स्ट्रक्चर डेटा को व्यवस्थित रूप से रखता है",
        "ऐरे समान प्रकार के तत्वों को संग्रहीत करता है",
        "लिंक्ड लिस्ट नोड्स से बनी होती है",
        "स्टैक लास्ट इन फर्स्ट आउट सिद्धांत का पालन करता है",
        "क्यू फर्स्ट इन फर्स्ट आउट सिद्धांत का पालन करती है",
    ],
    "Mixed": [
        "Data structures చాలా important",
        "Array లో elements store అవుతాయి",
        "Linked list లో nodes ఉంటాయి",
        "Stack follows LIFO principle",
        "Queue లో FIFO principle ఉపయోగిస్తారు",
    ],
}

# Combine all sentences for tokenizer training
all_sentences = []

for sentences in corpus.values():
    all_sentences.extend(sentences)

# Save the combined corpus
with open("corpus.txt", "w", encoding="utf-8") as file:
    for line in all_sentences:
        file.write(line + "\n")

print("Total training sentences:", len(all_sentences))


# Train a BPE tokenizer
bpe = Tokenizer(BPE(unk_token="[UNK]"))

bpe.train(
    ["corpus.txt"],
    BpeTrainer(
        vocab_size=500,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ),
)


# Train a WordPiece tokenizer
wordpiece = Tokenizer(WordPiece(unk_token="[UNK]"))
wordpiece.pre_tokenizer = Whitespace()

wordpiece.train(
    ["corpus.txt"],
    WordPieceTrainer(
        vocab_size=200,
        special_tokens=["[UNK]"],
    ),
)


# Train a SentencePiece Unigram tokenizer
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="sp",
    vocab_size=100,
    model_type="unigram",
    minloglevel=2,
)

sentencepiece_tokenizer = spm.SentencePieceProcessor()
sentencepiece_tokenizer.load("sp.model")


# Evaluate the tokenizers on each language
results = {
    "BPE": {},
    "WordPiece": {},
    "SentencePiece": {},
}

tokenizers = {
    "BPE": bpe,
    "WordPiece": wordpiece,
    "SentencePiece": sentencepiece_tokenizer,
}

for tokenizer_name, tokenizer in tokenizers.items():
    print("\nTOKENIZER:", tokenizer_name)

    for language, sentences in corpus.items():
        all_tokens = []

        for sentence in sentences:
            if tokenizer_name == "SentencePiece":
                tokens = tokenizer.encode_as_pieces(sentence)
            else:
                tokens = tokenizer.encode(sentence).tokens

            all_tokens.extend(tokens)

        total_tokens = len(all_tokens)

        results[tokenizer_name][language] = {
            "first_5_tokens": all_tokens[:5],
            "total_tokens": total_tokens,
        }

        print("\nLanguage:", language)
        print("First 5 Tokens:", all_tokens[:5])
        print("Total Token Count:", total_tokens)

    if tokenizer_name == "SentencePiece":
        vocab_size = tokenizer.get_piece_size()
    else:
        vocab_size = len(tokenizer.get_vocab())

    print("\nTotal Vocabulary Size:", vocab_size)
    results[tokenizer_name]["vocabulary_size"] = vocab_size