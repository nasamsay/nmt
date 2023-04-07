from functions import translate
class NMT:
    def predict(self, input):
        return translate.translate(input)
