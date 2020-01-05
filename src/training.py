import gensim
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logging.info("Starting Word2vec Training ...")

# Load the sentences/document strings containing words whose embeddings are to be learned.
pickle_file_path = ""
with open(pickle_file_path, "rb") as file:
    sentences = pickle.load(file)
logging.info(f"loaded {len(sentences)} sentences from file for training.")


sentences = [sent.split() for sent in a if len(sent.split())>3]
logging.info("Sentences split into words and cleaned.")
logging.info(f"New Length: {len(sentences)}")


# Build vocabulary
model = gensim.models.Word2Vec(
        sentences,
        size=300,
        min_count=20,
        workers=10)
logging.info("Gensim Model Loaded")

# Train model
model.train(sentences, total_examples=len(sentences), epochs=10)
logging.info("Gensim Model Trained")

# Save model
model.save('/output/trained_word2vec_model')
logging.info("Gensim Model Saved")

# Evaluate
model.accuracy('question_words.txt')

# Test the trained model manually

# find the most similar words to the given word
model.most_similar(positive=['India', 'New_Delhi'], negative=['Delhi'], topn=1)

# find the odd man out from the given words
model.doesnt_match("BJP AAP Congress Narendra_Modi".split())

# find the direct cosine similarity between two words
model.similarity('Delhi', 'India')

# find the most similar words to the given word learned by the model
logging.info("HIMA DAS")
model.most_similar('Hima_Das')

logging.info("Ben_stokes")
model.most_similar('Ben_Stokes')