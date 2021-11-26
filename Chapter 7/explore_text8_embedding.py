import gensim.downloader as api
import os
from gensim.models import KeyedVectors, Word2Vec

DATA = 'data'
WORD2VEC_BIN = f'{DATA}/text8-word2vec.bin'

dataset = api.load('text8')
model = Word2Vec(dataset)
if not os.path.exists(DATA):
    os.mkdir(DATA) 

if not os.path.exists(WORD2VEC_BIN):
    model.save(WORD2VEC_BIN)

def print_most_similar(word_conf_pairs, k):
    for i, (word, conf) in enumerate(word_conf_pairs):
        print("{:.3f} {:s}".format(conf, word))
        if i >= k-1:
            break
    if k < len(word_conf_pairs):
        print("...")



model = KeyedVectors.load(WORD2VEC_BIN)
word_vectors = model.wv

# get words in the vocabulary
words = list(word_vectors.index_to_key)
print([x for i, x in enumerate(words) if i < 10])
assert("king" in words)


print("# words similar to king")
print_most_similar(word_vectors.most_similar("king"), 5)

print("# vector arithmetic with words (cosine similarity)")
print("# france + berlin - paris = ?")
print_most_similar(word_vectors.most_similar(
    positive=["france", "berlin"], negative=["paris"]), 1
)

print("# vector arithmetic with words (Levy and Goldberg)")
print("# france + berlin - paris = ?")
print_most_similar(word_vectors.most_similar_cosmul(
    positive=["france", "berlin"], negative=["paris"]), 1
)

print("# find odd one out")
print("# [hindus, parsis, singapore, christians]")
print(word_vectors.doesnt_match(["hindus", "parsis", 
    "singapore", "christians"]))

print("# similarity between words")
for word in ["woman", "dog", "whale", "tree"]:
    print("similarity({:s}, {:s}) = {:.3f}".format(
        "man", word,
        word_vectors.similarity("man", word)
    ))

print("# similar by word")
print(print_most_similar(
    word_vectors.similar_by_word("singapore"), 5)
)

print("# distance between vectors")
print("distance(singapore, malaysia) = {:.3f}".format(
    word_vectors.distance("singapore", "malaysia")
))

vec_song = word_vectors["song"]
print("\n# output vector obtained directly, shape:", vec_song.shape)

vec_song_2 = word_vectors.word_vec("song", use_norm=True)
print("# output vector obtained using word_vec, shape:", vec_song_2.shape)
