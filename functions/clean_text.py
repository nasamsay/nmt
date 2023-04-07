from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def remove_punctuation(text):
    translator = str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\]^_{|}~")
    return text.translate(translator)

def clean_text(input):
    st_input = remove_punctuation(input)
    st_input= 'ST '+st_input.strip().lower()+' EN'
    with open('./functions/eng_tokenizer.pkl', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    st_input=eng_tokenizer.texts_to_sequences([st_input])
    st_input=pad_sequences(st_input, maxlen=17, padding='post')
    return st_input