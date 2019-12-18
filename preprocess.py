"""Preprocess data and build input pipeline."""
import unicodedata
import tensorflow as tf

FILE_PATH = "/Users/czar.yobero/datacsience/datasets/simpsons_scripts.txt"

def unicode_to_ascii(text):
    """Converts string to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def preprocess(file_path):
    """Returns list of formatted lines."""
    lines = open(file_path, 'r').read().lower().strip().split('<p>')
    lines = [unicode_to_ascii(line) for line in lines]
    lines = [lines[i].replace('</p>', '') for i in range(len(lines))]
    lines = ['<START>' + lines[i] + '<END>' for i in range(len(lines))]
    return lines


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

    tensor = line_tokenizer.texts_to_sequence(line)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')
    return tensor, line_tokenizer



def load_dataset(file_path):
    """Loads preprocessed dataset."""
    answer, question = create_dataset(file_path)
    question_tensor, question_tokenizer = tokenize(question)
    answer_tensor, answer_tokenizer = tokenize(answer)

    return question_tensor, answer_tensor, question_tokenizer, answer_tokenizer
