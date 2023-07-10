import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer

# Function to remove unnecessary characters and numbers from sentences


def remove(sentence):
    sentence = re.sub(r'[0-9]', ' ', sentence)
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    sentence = re.sub(r'[^A-Za-z\s]', '', sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('_', '')
    return sentence

# Function to tokenize sentences


def tokenize(sentence):
    return word_tokenize(sentence)

# Function to remove stopwords from tokens


def remove_stopwords(tokens):
    factory = StopWordRemoverFactory()
    stopwords_remover = factory.create_stop_word_remover()
    stopwords_dictionary = [
        'gak', 'masa', 'bisa', 'lagi', 'banget', 'sama', 'nya', 'saya', 'semua', 'kalo', 'saat', 'sambil', 'ya',
        'untuk', 'segitu', 'lain', 'sih', 'sangat', 'tidak', 'yang', 'tapi', 'itu', 'aduh', 'lah', 'buat', 'mah',
        'tahu', 'apa', 'mau', 'banyak', 'di', 'karena', 'bakal', 'padahal', 'ni', 'orang', 'terus', 'lain', 'sini',
        'hanya', 'dengan', 'aja', 'dan', 'ada', 'sekali', 'udh', 'kali', 'walaupun', 'pdhl', 'dari', 'cuma', 'juga',
        'sesuai', 'ini', 'jadi', 'tt'
    ]
    combined_stopwords = set(stopwords.words(
        'indonesian')) | set(stopwords_dictionary)
    filtered_tokens = [
        token for token in tokens if token.lower() not in combined_stopwords]
    return filtered_tokens
# Function to perform stemming on tokens


def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to preprocess the data


def preprocess_data(data):
    data['cleaning'] = data['Reviews'].apply(lambda x: remove(x))
    data['tokenization'] = data['cleaning'].apply(lambda x: tokenize(x))
    data['stopwords'] = data['tokenization'].apply(
        lambda x: remove_stopwords(x))
    data['stemming'] = data['stopwords'].apply(lambda x: stemming(x))
    return data

# Function to train the model and make predictions


def train_model(data):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Convert the preprocessed data to text representation
    text_data = preprocessed_data['stemming'].apply(
        lambda tokens: ' '.join(tokens))

    # Create the CountVectorizer object
    vectorizer = CountVectorizer()

    # Transform the data into term frequency matrix
    tf_matrix = vectorizer.fit_transform(text_data)

    # Get the list of features (unique words)
    features = vectorizer.get_feature_names_out()

    # Create a DataFrame from the term frequency matrix
    df_tf = pd.DataFrame(tf_matrix.toarray(), columns=features)

    # Calculate chi-square scores
    labels = preprocessed_data['Label']
    chi2_scores = chi2(df_tf, labels)[0]

    # Select the top features based on chi-square scores
    total_features = df_tf.shape[1]
    k = int(0.5 * total_features)
    selector = SelectKBest(chi2, k=k)
    selected_matrix = selector.fit_transform(df_tf, labels)
    selected_features_indices = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_features_indices]
    df_selected_features = pd.DataFrame(
        selected_matrix, columns=selected_features)
    df_selected_features['Label'] = labels

    # Split the data into train and test sets
    X = df_selected_features
    y = preprocessed_data['Label'].apply(
        lambda x: 'negatif' if x == 0 else 'positif')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40)

    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    return y_pred

# Create the Streamlit app


def main():
    st.title("Aplikasi Analisis Sentimen")
    st.write("Aplikasi Sistem Analisis Sentimen Dibangun Dengan metode Naive Bayes")

    # File upload widget
    st.subheader("Silahkan Upload File Anda")
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return

        # Define stopwords dictionary
        stopwords_dictionary = [
            'gak', 'masa', 'bisa', 'lagi', 'banget', 'sama', 'nya', 'saya', 'semua', 'kalo', 'saat', 'sambil', 'ya',
            'untuk', 'segitu', 'lain', 'sih', 'sangat', 'tidak', 'yang', 'tapi', 'itu', 'aduh', 'lah', 'buat', 'mah',
            'tahu', 'apa', 'mau', 'banyak', 'di', 'karena', 'bakal', 'padahal', 'ni', 'orang', 'terus', 'lain', 'sini',
            'hanya', 'dengan', 'aja', 'dan', 'ada', 'sekali', 'udh', 'kali', 'walaupun', 'pdhl', 'dari', 'cuma', 'juga',
            'sesuai', 'ini', 'jadi', 'tt'
        ]

        # Train the model and make predictions
        if st.button("Analisis Sentiment"):
            predictions = train_model(data)

            # Display the data and predictions
            st.subheader("Data")
            st.write(data)

            st.subheader("Predictions")
            st.write(predictions)

            # Menghitung total sentimen
            total_negatif = (predictions == 'negatif').sum()
            total_positif = (predictions == 'positif').sum()

            st.header('Summary')
            st.write("Total Sentimen Positif: ", total_positif)
            st.write('Total Sentimen Negatif: ', total_negatif)


if __name__ == '__main__':
    main()
