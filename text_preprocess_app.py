import streamlit as st
import io
import nltk
import string
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from docx import Document
# nltk.download('wordnet')  

# preprocessing the text  - removing stopwords, punctuation, lemmatization
st.cache_data
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    #stemmer = nltk.PorterStemmer()
    # Convert text to lowercase
    tokens = nltk.word_tokenize(text.lower())  
    # Remove stopwords and lemmatize
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stopwords]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove punctuation
    # stemmed_tokens = [nltk.PorterStemmer().stem(token) for token in tokens]
    lemmatized_tokens_no_punct = [''.join(c for c in token if c not in string.punctuation) for token in lemmatized_tokens]
    stemmed_tokens_no_punct = [token for token in lemmatized_tokens_no_punct if token]
    return ' '.join(lemmatized_tokens_no_punct)

# Function to extract keywords from text using the model
st.cache_data
def extract_keywords(text, n):
    preprocessed_text = preprocess(text)
    
    # can use pre trained model like BERT, GPT-2, etc. for keyword extraction
    vectorizer = TfidfVectorizer() #ngram_range=(1, 2) for bigrams 
    # Fit and transform the pre rocessed text
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    
    feature_names = vectorizer.get_feature_names_out()
    #Get the TF-IDF scores for each feature
    tfidf_scores = tfidf_matrix.toarray()[0]#.flatten()
    # Create a dictionary of feature names and their corresponding TF-IDF scores
    tfidf_dict = dict(zip(feature_names, tfidf_scores))
    #sort the dictionary by TF-IDF scores in descending order
    sorted_tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))
    keywords = list(sorted_tfidf_dict.keys())[:n]
    return keywords

# Streamlit UI
st.title('Keyword Extraction App')

st.info("Please upload a Text (.txt)/ PDF (.pdf)/ Docs (.docx) file for keyword extraction.")

# File upload
uploaded_file = st.file_uploader("Upload a document", type=['txt', 'pdf','docx'])


if uploaded_file is not None:
    
    # User input for number of keywords
    num_keywords = st.slider('Number of keywords to extract', 1, 20, 5)
    
    #if the user want to manually enter the number of keywords
    # num_keywords = st.number_input('Enter the number of keywords to extract', min_value=1, value=5, max_value=25)

    
    # Read the uploaded file as a string
    if uploaded_file.type == 'text/plain':
        text = uploaded_file.getvalue().decode("utf-8")
        
    elif uploaded_file.type == 'application/pdf':
        text = ""
        try:
            # Use PyPDF2 to process the PDF file into text
            pdf_file_upload = io.BytesIO(uploaded_file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_file_upload)
            text = " ".join(page.extract_text() for page in pdf_reader.pages)
            pdf_file_upload.close()
        except:
            st.error("Error reading PDF file. Please make sure the file is not corrupted or encrypted.")
            st.stop()
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            # Use python-docx to process the .docx file into text
            doc_file_upload = io.BytesIO(uploaded_file.read())
            doc = Document(doc_file_upload)
            text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        except:
            st.error("Error reading .docx file. Please make sure the file is not corrupted.")
            st.stop()
    else:
        st.error("Unsupported file format. Please upload a .txt or .pdf or a .docx file.")
        st.stop()
    
    # Extract keywords
    keywords = extract_keywords(text, num_keywords)
        
    # Display extracted keywords
    st.write('Keywords:', ', '.join(keywords))
    
    
    # Downloadable results
    keywords_str = ', '.join(keywords)
    st.download_button(
        label="Download keywords",
        data=keywords_str,
        file_name='keywords.txt',
        mime='text/plain',)