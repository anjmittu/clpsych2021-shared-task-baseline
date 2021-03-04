import argparse
import os
import json
from spacy.lang.en import stop_words
import twikenizer as twk
import re
import string
from nltk.corpus import words


spacy_stopwords = stop_words.STOP_WORDS
tokenizer = twk.Twikenizer()

word_dictionary = list(set(words.words()))
for alphabet in "bcdefghjklmnopqrstuvwxyz":
    word_dictionary.remove(alphabet)

def append_json(json_data, file_path):
    """
    Write json to file
    :param json_data: Json to write
    :param file_path: Path of the file
    """
    with open(file_path, 'a+') as f:
        json.dump(json_data, f)
        f.write("\n")

def split_hashtag_to_words_all_possibilities(hashtag):
    all_possibilities = []

    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]
    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)

            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]

    return all_possibilities

def split_hashtag(token):
    split_hashtag = re.findall('[A-Z][^A-Z]*', token)
    if len(split_hashtag) > 1:
        return split_hashtag
    split_hashtag = token.split('_')
    if len(split_hashtag) > 1:
        return split_hashtag
    all_possibilities = split_hashtag_to_words_all_possibilities(token)
    min_split = float("inf")
    for pos in all_possibilities:
        if len(pos) < min_split:
            min_split = len(pos)

    for pos in all_possibilities:
        if len(pos) == min_split:
            return pos

    return ""


def normalizeToken(token):
    lowercase_token = token.lower()
    if token.startswith("@"):
        return ""
    elif lowercase_token.startswith("http") or lowercase_token.startswith("www"):
        return ""
    elif token.startswith("#"):
        return split_hashtag(token[1:])
    elif token in string.punctuation:
        return ""
    elif len(token) == 1:
        return ""
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def tokenize_tweets(tweets):
    new_tweets = []
    for tweet in tweets:
        tweet_tokens = []
        for token in tokenizer.tokenize(re.sub(r'http\S+', '', tweet.lower())):
            norm_token = normalizeToken(token)
            if isinstance(norm_token, list):
                for t in norm_token:
                    if t not in spacy_stopwords:
                        tweet_tokens.append(t)
            else:
                if token not in spacy_stopwords and token != "":
                    tweet_tokens.append(norm_token)
        new_tweets.append(tweet_tokens)
    return new_tweets


def tokenize_file(file_path, save_file_path):
    with open(file_path, 'r') as f:
        for json_obj in f:
            data_json = json.loads(json_obj)
            raw_tweets = []
            for tweet in data_json["tweets"]:
                raw_tweets.append(tweet["text"])
            tweets = tokenize_tweets(raw_tweets)
            user_json = {"text": tweets,
                         "id":data_json["id"]}
            if "label" in data_json:
                user_json["label"] = data_json["label"]
            append_json(user_json, save_file_path)

def main(input_file_path, output_file_path):
    training_file_path = os.path.join(output_file_path, "train_tokenized.jsonl")
    testing_file_path = os.path.join(output_file_path, "test_tokenized.jsonl")

    if os.path.exists(training_file_path):
        os.remove(training_file_path)
    if os.path.exists(testing_file_path):
        os.remove(testing_file_path)

    print("Tokenizing training data")
    tokenize_file(os.path.join(input_file_path, "train.jsonl"), training_file_path)
    if os.path.exists(os.path.join(input_file_path, "test.jsonl")):
        print("Tokenizing test data")
        tokenize_file(os.path.join(input_file_path, "test.jsonl"), testing_file_path)
    else:
        print("No test data found")
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizes the data for the shared task')
    parser.add_argument('--input', help='the directory with the input files', type=str)
    parser.add_argument('--output', help='the directory where the output files should go',
                        type=str)
    args = parser.parse_args()
    main(args.input, args.output)
