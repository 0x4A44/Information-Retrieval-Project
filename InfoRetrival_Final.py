"""
Jeudy Diaz

Information Retrieval: Final Project

*Large Data and Zone Indexing*

Zone Encoded in the posting list eg. "egg" - > [(doc1: title, doc1:abstract), (doc2:title), (doc3:abstract)]
"""
import json
import re
from csv import DictReader
import joblib
import math
from nltk.corpus import stopwords
import time


class Appearance:
    def __init__(self, docId, frequency):
        self.docId = docId
        self.frequency = frequency

    def __repr__(self):
        return str(self.__dict__)


class Database:
    def __init__(self):
        self.db = dict()

    def __repr__(self):
        return str(self.__dict__)

    def get(self, id):
        return self.db.get(id, None)

    def add(self, document):
        if type(document) is dict:
            return self.db.update({document['Id']: document})

    def remove(self, document):
        return self.db.pop(document['Id'], None)


def get_words_in_document(file):
    words_in_title = {}
    words_in_abstract ={}
    title_words= 0
    abstract_words = 0
    for line in file:
        json_data = json.loads(line)
        id = json_data['pmid']
        for words in json_data['title']:
            title_words += 1
        words_in_title[id] = title_words
        title_words = 0
        for words in json_data['abstract']:
            abstract_words += 1
        words_in_abstract[id] = abstract_words
        abstract_words = 0
    return words_in_title, words_in_abstract


class InvertedIndex:

    def __init__(self, db):
        self.index = dict()
        self.db = db
        self.temp_list = []

    def __repr__(self):
        return str(self.index)

    def inverted_index(self, file):
        stop_words = set(stopwords.words('English'))
        for line in file:
            json_data = json.loads(line)
            id = json_data['pmid']
            title_appearances_dict = dict()
            clean_title_text = re.sub(r'[^\w\sa-zA-Z]', '', json_data['title'])
            title_words = clean_title_text.split(' ')
            for word in title_words:
                if word in stop_words:
                    title_words.remove(word)
            abstract_appearances_dict = dict()
            clean_abstract_text = re.sub(r'[^\w\sa-zA-Z]', '', json_data['abstract'])
            abstract_words = clean_abstract_text.split(' ')
            for word in abstract_words:
                if word in stop_words:
                    abstract_words.remove(word)
            for term in title_words:
                term = term.lower()
                term_frequency = title_appearances_dict[term]['title'] if term in title_appearances_dict else 0
                title_appearances_dict[term] = {'title': term_frequency + 1}
            for term in abstract_words:
                term = term.lower()
                term_frequency = abstract_appearances_dict[term]['abstract'] if term in abstract_appearances_dict else 0
                abstract_appearances_dict[term] = {'abstract': term_frequency + 1}
            combined_appearances_dict = dict()
            if len(title_appearances_dict) < len(abstract_appearances_dict):
                for term in title_appearances_dict.keys():
                    if term in abstract_appearances_dict.keys():
                        combined_appearances_dict[term] = {id: [title_appearances_dict[term], abstract_appearances_dict[term]]}
                    else:
                        combined_appearances_dict[term] = {id: [title_appearances_dict[term]]}
                for term in abstract_appearances_dict.keys():
                    if term not in combined_appearances_dict.keys():
                        combined_appearances_dict[term] = {id: [abstract_appearances_dict[term]]}
            else:
                for term in abstract_appearances_dict.keys():
                    if term in title_appearances_dict.keys():
                        combined_appearances_dict[term] = {id: [title_appearances_dict[term], abstract_appearances_dict[term]]}
                    else:
                        combined_appearances_dict[term] = {id: [abstract_appearances_dict[term]]}
                for term in title_appearances_dict.keys():
                    if term not in combined_appearances_dict.keys():
                        combined_appearances_dict[term] = {id: [title_appearances_dict[term]]}
            update_dict = {key: [appearance] if key not in self.index else self.index[key] + [appearance] for (key, appearance) in combined_appearances_dict.items()}
            self.index.update(update_dict)
        joblib.dump(self.index, 'inverted_index1.txt')

    def index_keys(self):
        dict_keys = []
        for key in self.index.keys():
            dict_keys.append(key)
        return dict_keys.__repr__()

    def amount_of_terms(self):
        term_amount = 0
        for key in self.index.keys():
            term_amount += 1
        return term_amount

    def index_filter(self):
        stop_words = set(stopwords.words('English'))
        for word in list(self.index.keys()):
            if word in stop_words:
                del self.index[word]

    def merge_algorithm(self):
        file = open('data.txt', 'r', encoding="utf8")
        words_in_title, words_in_abstract = get_words_in_document(file)
        file.close()
        queries = True
        while queries:
            query_posting_list = []
            amount_of_words = input("How many words would you like to search for (Max is 2): ")
            if amount_of_words == '2':
                word1 = input("Enter a word to search for: ")
                word1_appearance = input("Would you like this word to appear in the title or abstract: --> ")
                word2 = input("Enter a word to search for: ")
                word2_appearance = input("Would you like this word to appear in the title or abstract: --> ")
                try:
                    if word1_appearance == 'title':
                        word1_posting = self.tf_idf_posting(words_in_title, word1, word1_appearance)
                    else:
                        word1_posting = self.tf_idf_posting(words_in_abstract, word1, word1_appearance)
                    if word2_appearance == 'title':
                        word2_posting = self.tf_idf_posting(words_in_title, word2, word2_appearance)
                    else:
                        word2_posting = self.tf_idf_posting(words_in_abstract, word2, word2_appearance)
                except KeyError or ZeroDivisionError:
                    print("Word not found")
                print("POSTING LIST FOR",word1_appearance + ":", word1,"is: ",word1_posting,"\n" )
                print("POSTING LIST FOR",word2_appearance + ":", word2,"is: ",word2_posting,"\n")

                if len(word1_posting) > len(word2_posting):
                    for key, score in word2_posting.items():
                        if key in word1_posting:
                            query_posting_list.append(key)
                else:
                    for key, score in word1_posting.items():
                        if key in word2_posting:
                            query_posting_list.append(key)

                print("POSTING LIST FOR", word1_appearance + ":", word1, "AND", word2_appearance + ":", word2, "is:", query_posting_list)

            else:
                word = input("Enter a word to search for: ")
                word_appearance = input("Would you like this word to appear in the title: or abstract: --> ")

                try:
                    if word_appearance == 'title':
                        word_posting = self.tf_idf_posting(words_in_title, word, word_appearance)
                    else:
                        word_posting = self.tf_idf_posting(words_in_abstract, word, word_appearance)
                except KeyError or ZeroDivisionError:
                    print("Word not found")

                    for key in word_posting.keys():
                        query_posting_list.append(key)

                print("\nPOSTING LIST FOR", word_appearance + ":", word, "is: ", word_posting, "\n")
            contin = input("\n Would you like to search for more terms? (Y or N)")
            if contin == 'N':
                queries = False

    def tf_idf_posting(self, word_list, word, position):
        doc_tf_idf = {}
        total_docs = 104879
        word_posting_list = self.get_posting(word, position)
        word_posting = []
        idf_score = math.log10(total_docs / len(word_posting_list))
        for item in word_posting_list:
            for key in item.keys():
                word_posting.append(key)
        for item in word_posting_list:
            for key, zones in item.items():
                for zone in zones:
                    if position in zone:
                        tf = zone[position] / word_list[key]
                        idf = idf_score
                        doc_tf_idf[key] = tf * idf
        sorted_posting_list = {}
        sorted_keys = sorted(doc_tf_idf, key=doc_tf_idf.get, reverse=True)
        for key in sorted_keys:
            sorted_posting_list[key] = doc_tf_idf[key]
        return sorted_posting_list

    def get_posting(self, word, position):
        posting_list = []
        full_posting = self.index[word]
        for id_dict in full_posting:
            for id in id_dict.values():
                for zone in id:
                    if position in zone:
                        posting_list.append(id_dict)
        return posting_list


if __name__ == '__main__':
    #Data.txt holds the raw information that is used to create the inverted index. This takes a very long time.:
    #file = open('data.txt', 'r', encoding="utf8")

    my_db = Database()
    my_inverted_index = InvertedIndex(my_db)
    start_time = time.time()
    print("Initializing inverted index. This may take a few minutes...")
    my_inverted_index.index = joblib.load('inverted_index1.txt')
    #my_inverted_index.inverted_index(file)
    #file.close()
    elapsed_time = time.time() - start_time
    print("Load Inverted Index File Time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    my_inverted_index.merge_algorithm()


