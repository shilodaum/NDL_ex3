import nltk
import spacy
import matplotlib.pyplot as plt
import numpy as np
import string
import regex
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

punctuations = tuple("!\"#“$%&'”‘’()*+, -./:;<=>?@[\\]^_`{|}~")


def read(filename, encoding='utf-8'):
    """
    read file from utf-8 encoding
    :param filename: file path
    :return: file text
    """
    with open(filename, 'r', encoding=encoding) as f:
        return f.read()


def tokenize(text):
    """
    tokenize words
    :param text: text
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    return tokens


def count_occurrences(tokens):
    """
    count occurrences
    :param tokens: list of tokens
    :return: counter list of tuples: token and count
    """
    counter = dict()

    for key in tokens:
        if key in counter:
            counter[key] += 1
        else:
            counter[key] = 1
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return sorted_counter


def plot_log_freq(counter, figure_name):
    """
    plot the rankand log frequency
    :param counter: token counter
    :param figure_name: figure name
    """
    values = [value for key, value in counter]
    plt.plot(values)
    plt.yscale('log', base=10)
    plt.title(f'Tokens in the book: {figure_name}')
    plt.xlabel('rank')
    plt.ylabel('frequency (log scale)')
    plt.savefig(f'{figure_name}.png')
    plt.show()


def filter_stopwords(tokens, stop_words, others_to_remove=tuple()):
    """
    filter out all stopwords
    :param tokens: set of tokens
    :param stop_words: list of english stopwords
    :param others_to_remove: list of english unwanted_words

    :return: filtered tokens
    """
    filtered_tokens = []
    for w in tokens:
        if w.lower() not in stop_words and w.lower() not in others_to_remove:
            filtered_tokens.append(w)
    return filtered_tokens


def filter_counter(counter, filter_tokens):
    """
    filter the counter according to the filter_tokens
    :param counter: counter with all tokens
    :param filter_tokens: the wanted tokens to take
    :return:
    """
    return [(key, value) for key, value in counter if key in filter_tokens]


def print_top_values(counter, num=20):
    """
    print the top 20 tokens
    :param counter: counter with all tokens
    :param num: number of values to print
    """
    print(f'Top {num} words:')
    for token, count in counter[:num]:
        print(f'{token} - {count}')


def part_b(text):
    tokens = tokenize(text)
    counter = count_occurrences(tokens)
    plot_log_freq(counter, 'log frequency')
    # print_top_20(counter)
    return tokens, counter


def part_c(tokens, counter):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = filter_stopwords(set(tokens), stop_words)
    filtered_counter = filter_counter(counter, filtered_tokens)
    plot_log_freq(filtered_counter, 'filtered stopwords')
    # print_top_20(filtered_counter)
    return filtered_tokens, filtered_counter


def part_c_punctuations(tokens, counter):
    filtered_tokens = filter_stopwords(set(tokens), list(), punctuations)
    filtered_counter = filter_counter(counter, filtered_tokens)
    plot_log_freq(filtered_counter, 'filtered stopwords and punctuations')
    # print_top_20(filtered_counter)
    return filtered_tokens, filtered_counter


def part_d(tokens, counter):
    ps = PorterStemmer()
    dcounter = dict()
    for key, value in counter:
        stem_key = ps.stem(key)
        if stem_key in dcounter:
            dcounter[stem_key] += value
        else:
            dcounter[stem_key] = value
    filtered_counter = sorted(dcounter.items(), key=lambda x: x[1], reverse=True)
    filtered_tokens = list(dict(filtered_counter).keys())
    # filtered_counter = filter_counter(counter, filtered_tokens)
    plot_log_freq(filtered_counter, 'stemmed')
    print_top_values(filtered_counter)
    return filtered_tokens, filtered_counter


def part_e(text):
    """
    run POS tagging and extract adj+noun tokens

    :return:
    """
    pos_tags = nltk.pos_tag(tokenize(text))
    # print(type(tags))
    # print(pos_tags[:100])
    tokens = list()
    token = ''
    prev_pos = 'No'
    for word, pos in pos_tags:
        if prev_pos == 'No':
            if pos in ['JJ', 'JJS', 'JJR'] and word not in punctuations:
                prev_pos = 'Adj'
                token += word

        elif prev_pos == 'Adj':
            if pos in ['JJ', 'JJS', 'JJR'] and word not in punctuations:
                token += ' ' + word

            elif pos in ['NN', 'NNS', 'NNP', 'NNPS'] and word not in punctuations:
                prev_pos = 'Noun'
                token += ' ' + word
            else:
                token = ''
                prev_pos = 'No'

        elif prev_pos == 'Noun':
            if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and word not in punctuations:
                token += ' ' + word
            else:
                tokens.append(token)
                token = ''
                if pos in ['JJ', 'JJS', 'JJR'] and word not in punctuations:
                    prev_pos = 'Adj'
                    token += word
                else:
                    prev_pos = 'No'

    counter = count_occurrences(tokens)

    plot_log_freq(counter, 'adj+noun')
    print_top_values(counter)
    return tokens, counter


def part_f(text):
    pos_tags = nltk.pos_tag(tokenize(text))
    print(pos_tags)
    # Finding manually...
    # Found!
    # ... and said in the gentlest of accents:


def part_g(text):
    pos_tags = nltk.pos_tag(tokenize(text))
    pos_types_counter = dict()

    for token, pos in pos_tags:
        if token not in punctuations:
            if token in pos_types_counter:
                if pos in pos_types_counter[token]:
                    pos_types_counter[token][pos] += 1
                else:
                    pos_types_counter[token][pos] = 1
            else:
                pos_types_counter[token] = dict()
                pos_types_counter[token][pos] = 1
    counter_pos = [(key, list(pos_dict.keys())) for key, pos_dict in pos_types_counter.items()]

    sorted_pos_counter = sorted(counter_pos, key=lambda x: len(x[1]), reverse=True)
    # print(sorted_pos_counter)

    print_top_values(sorted_pos_counter, 10)
    print_top_values(list(reversed(sorted_pos_counter)), 10)
    return sorted_pos_counter


def part_h(text):
    pos_tags = nltk.pos_tag(tokenize(text))
    new_text = ''
    for word, pos in pos_tags:
        if pos in ['NNP', 'NNPS']:
            new_text += ' ' + word
    with open('proper nouns text.txt', 'w', encoding='utf-8') as f:
        f.write(new_text)


def part_i(text):
    regex_expr = r'\b([a-zA-Z]\w*)[\s!\"#$%&\'()*+,.<>=\?@;\-\^_{}~]+\1\b'
    for m in regex.finditer(regex_expr, text):
        start = m.start()
        end = m.end()
        #print(text[start:end])
        print(m.group())


def main():
    text = read('pride and prejudice.txt')
    # tokens, counter = part_b(text)
    # f_tokens, f_counter = part_c(tokens, counter)
    # fp_tokens, fp_counter = part_c_punctuations(f_tokens, f_counter)
    # fps_tokens, fps_counter = part_d(fp_tokens, fp_counter)
    # tokens = part_e(text)
    # pos_counter = part_g(text)
    part_i(text)
    # print(pos_counter)


if __name__ == '__main__':
    main()
