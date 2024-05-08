import streamlit as st
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from sklearn.feature_extraction.text import TfidfVectorizer




def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made ",
        " with ❤️ by ",
        "Jaydeep karode, Krishna mandloi, madhur dubey, krishnapal songara",
        br(),
        link("https://aitr.ac.in/", image('https://acropolis.in/wp-content/uploads/2023/03/unnamed-1024x203.png',width=px(90), height=px(40))),
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()
ps = PorterStemmer()

##start
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # tfidf = pickle.load(open('vectorizer.pkl','rb'))
# import pickle

# try:
#     # Attempt to load the pickled object
#     with open('vectorizer.pkl', 'rb') as f:
#         tfidf = pickle.load(f)
# except FileNotFoundError:
#     # Handle the case where the file doesn't exist
#     print("Error: 'vectorizer.pkl' not found. Please ensure the file exists.")
#     # You can add further logic here, such as creating the file or its content.
# except Exception as e:
#     # Handle other exceptions gracefully
#     print("An error occurred while loading the pickled object:", e)
# # model = pickle.load(open('model.pkl','rb'))

# import pickle

# # Handle vectorizer.pkl
# try:
#     with open('vectorizer.pkl', 'rb') as f:
#         tfidf = pickle.load(f)
# except FileNotFoundError:
#     print("Error: 'vectorizer.pkl' not found. Please ensure the file exists.")
#     # Handle this case appropriately, such as creating the file or its content.

# # Handle model.pkl
# try:
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
# except FileNotFoundError:
#     print("Error: 'model.pkl' not found. Please ensure the file exists.")
#     # Handle this case appropriately, such as creating the file or its content.



# st.title("Email/SMS Spam Classifier")
# st.header('By :blue[_Jaydeep Karode, krishna mandloi, krishnapal and madhur_] :sunglasses:', divider='rainbow')
# # st.header('_Streamlit_ is :blue[cool] :sunglasses:')

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # Preprocess the input text
#     transformed_sms = transform_text(input_sms)
#     # Vectorize the preprocessed text using TF-IDF
#     vector_input = tfidf.transform([transformed_sms])
#     # Predict using the model
#     result = model.predict(vector_input)[0]
#     # Display the result
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
# ##end

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load TF-IDF vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' not found. Please ensure the file exists.")
    st.stop()  # Stop execution if file not found
except Exception as e:
    st.error("An error occurred while loading the pickled object:", e)
    st.stop()  # Stop execution if an error occurs during loading

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the file exists.")
    st.stop()  # Stop execution if file not found
except Exception as e:
    st.error("An error occurred while loading the pickled object:", e)
    st.stop()  # Stop execution if an error occurs during loading

# Main Streamlit app
st.title("Email/SMS Spam Classifier")
st.header('By :blue[_Jaydeep Karode, krishna mandloi, krishnapal and madhur_] :sunglasses:', divider='rainbow')
# st.header('_Streamlit_ is :blue[cool] :sunglasses:')

# Input text area
input_sms = st.text_area("Enter the message")

# Predict button click event
if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed text using TF-IDF
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the model
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")