import pickle
import sklearn
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import  wordpunct_tokenize
import string
from nltk.stem import PorterStemmer
import warnings
import gradio as gr

warnings.filterwarnings("ignore")

model = pickle.load(open('model.pkl', 'rb'))
countvect = pickle.load(open('CountVectorizer.pkl', 'rb'))

def convert(text):
    text = unidecode(text.lower())
    stops = stopwords.words('english')
    y = []
    for i in wordpunct_tokenize(text):
        if (i not in stops and i not in string.punctuation):
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if (i.isalnum()):
            y.append(i)

    text = y.copy()
    y.clear()
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

def sms_predict(text):
    processed_text = convert(text)
    vect_text = countvect.transform([processed_text]).toarray()
    value = model.predict_proba(vect_text)
    return {"Not Spam":value[0][0], "Spam":value[0][1]}

interface =  gr.Interface(sms_predict,
                          inputs='text',
                          outputs='label',
                          title="Email/SMS Spam Detection",
                          description="Checks the messages you got in mails/sms are spam or not.",
                          examples=[
                              ["Hey! We're excited to share a special discount with our loyal customers. Use code SAVE15 for extra savings on your next purchase."],
["Hi there! Your package is out for delivery and will arrive by 3 PM. Track your shipment here: [tracking link]."],
                              ["Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo. Not that i'm trying to invite myself or anything!"],
                                    ["Congratulations! You've won a luxury vacation! Click the link to claim your prize now. Limited time offer!"]

]
                          )

if __name__ == "__main__":
    interface.launch(share=True, show_api=False)

print("Hello World")
