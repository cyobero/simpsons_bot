"""Preprocess data and build input pipeline."""
import unicodedata
import re
import tensorflow as tf

FILE_PATH = "./simpsons_scripts_3.txt"

def unicode_to_ascii(text):
    """Converts string to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def preprocess(file_path):
    """Returns list of formatted lines."""
    lines = open(file_path, 'r').read().lower().strip().split('<p>')
    lines = [unicode_to_ascii(line) for line in lines]
    lines = [lines[i].replace('</p>', '') for i in range(len(lines))]

    # Create space between words and punctuations after it.
    # EX: "he is a boy !" ----> "he is a boy !"
    lines = [re.sub(r"([?.!,¿])", r" \1 ", line) for line in lines]
    lines = [re.sub(r'[" "]+', " ", line) for line in lines]

    # Replace everything with spaces except (a-Z, A-Z, "?", "!", ".", ",")
    lines = [re.sub(r"[^a-zA-Z?.!,¿]+", " ", line) for line in lines]

    lines = ['<start>' + lines[i] + '<end>' for i in range(len(lines))]
    return lines[1: ]


def create_dataset(file_path):
    """Returns question/answer pair in format: [QUESTION, ANSWER]"""
    lines = preprocess(file_path)

    # Assign each line i to questions and each line i+1 to answers that corresponds
    # to question i.
    questions, answers = [
        lines[i] for i in range(len(lines)-1)], [lines[i+1] for i in range(len(lines)-1)]
    return questions, answers


def max_length(tensor):
    """Returns max length of tensor"""
    return max(len(t) for t in tensor)


def tokenize(line):
    """Tokenizes line."""
    line_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    line_tokenizer.fit_on_texts(line)

    tensor = line_tokenizer.texts_to_sequences(line)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')

    # Ensure that we return a nonzero tensor to reduce dimensionality and
    # improve performance. 
    tensor = tensor[tensor.nonzero()]
    tensor.resize((len(tensor), 11), refcheck=False)
    return tensor, line_tokenizer



def load_dataset(file_path):
    """Loads preprocessed dataset."""
    answer, question = create_dataset(file_path)
    question_tensor, question_tokenizer = tokenize(question)
    answer_tensor, answer_tokenizer = tokenize(answer)

    return question_tensor, answer_tensor, question_tokenizer, answer_tokenizer
