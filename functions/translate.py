from tensorflow.keras.models import load_model
from functions import clean_text
import pickle
import numpy as np
encoder = load_model('./functions/encoder.h5')
sampler = load_model('./functions/sampler.h5')
with open('./functions/fr_tokenizer.pkl', 'rb') as handle:
    fr_tokenizer = pickle.load(handle)
def translate(input):
    st_input = clean_text.clean_text(input)
    st_h, st_c = encoder.predict(st_input)

    st_input = fr_tokenizer.word_index['st']
    st_input = np.array([[st_input]])  

    prediction_tok = []                     
    for i in range(346):
        probs, st_h, st_c = sampler.predict([st_input, st_h, st_c])
        
        st_input = probs.argmax(axis=-1)
        
        token = probs.argmax()
        if token != fr_tokenizer.word_index['en']:
            prediction_tok.append(token)
        
        if token == fr_tokenizer.word_index['en']:

            break    

    words = [fr_tokenizer.index_word[x] for x in prediction_tok if x in fr_tokenizer.index_word]
    words=' '.join(words)
    return words

