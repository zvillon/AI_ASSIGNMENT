import unicodedata
import re


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1


def read_data(lang1, lang2, reverse=False):
    try:
        filename = "fra.txt"
        print(f"Reading data file: {filename}")
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Data file '{filename}' not found.")
        print("Please download 'fra-eng.zip' from http://www.manythings.org/anki/")
        print(
            "Unzip it, and place the resulting 'fra.txt' file in the same directory as this script."
        )
        print("---------------")
        exit()

    pairs = [[normalize_string(s) for s in l.split("\t")[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang, output_lang = Vocabulary(lang2), Vocabulary(lang1)
    else:
        input_lang, output_lang = Vocabulary(lang1), Vocabulary(lang2)
    return input_lang, output_lang, pairs


def filter_pairs(pairs, max_length=10):
    return [
        p
        for p in pairs
        if len(p[0].split(" ")) < max_length and len(p[1].split(" ")) < max_length
    ]
