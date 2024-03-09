import re
import spacy
from spacy.matcher import Matcher
from transformers import pipeline
import pandas as pd
import streamlit as st

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
)

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
roberta_sentiment_classifier = pipeline(
    "sentiment-analysis", model=model_path, tokenizer=model_path
)

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

# Initialize Matcher with the current NLP vocab
matcher = Matcher(nlp.vocab)

# Define pattern for emails and a general pattern for phone numbers
matcher.add("EMAIL", [[{"LIKE_EMAIL": True}]])

# List of common interrogative words in Portuguese
interrogative_words = [
    "quem",
    "qual",
    "quais",
    "quando",
    "onde",
    "como",
    "por que",
    "para que",
]

# List of imperative verbs indicating a command (example list, needs expansion)
command_verbs = [
    "faça",
    "venha",
    "use",
    "leia",
    "escreva",
    "envie",
    "fale",
]

ignore_tokens = [
    "é",
    "são",
    "será",
    "seriam",
    "o",
    "a",
    "os",
    "as",
    "seu",
    "sua",
    "seus",
    "suas",
]

def read_conversation_file(file_path):
    """Reads content from a specified file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        txt = file.read()
    return txt

def extract_actor_and_sentence(line):
    """Extracts the actor and the sentence from a line, identifying speaker prefixes."""
    # Identify speaker prefixes and extract the sentence
    match = re.match(r"^(Entrevistador|Pessoa):\s*(.*)", line.strip())
    if match:
        actor = match.group(1)  # Actor (Entrevistador or Pessoa)
        sentence = match.group(2).strip()  # The actual sentence
    else:
        actor = "Unknown"  # Fallback actor if no match
        sentence = line.strip()  # The original line as the sentence

    return actor, sentence


def find_phone_numbers(text):
    # Define a regex pattern for phone numbers
    # This pattern aims to match phone numbers in the formats you described
    phone_pattern = r"\(?\d{2}\)?[-\s]?\d{4,5}[-\s]?\d{4}"
    matches = re.findall(phone_pattern, text)
    return matches


def sentiment_analysis(sentence):
    # DistilBERT analysis
    distilbert_scores = distilled_student_sentiment_classifier(sentence)[0]
    distilbert_sentiment_map = {
        item["label"]: item["score"] for item in distilbert_scores
    }
    distilbert_dominant_sentiment = max(
        distilbert_sentiment_map, key=distilbert_sentiment_map.get
    )

    # Roberta analysis
    roberta_scores = roberta_sentiment_classifier(sentence)[0]
    roberta_dominant_sentiment = roberta_scores["label"]

    return distilbert_dominant_sentiment, roberta_dominant_sentiment


def extract_subject_question(sentence):
    """Extracts the subject from a question sentence, focusing on tokens following interrogative words."""
    doc = nlp(sentence)
    subject = ""
    found_interrogative = False
    temp_tokens = []

    for token in doc:
        # Check for interrogative words using the correct attribute
        if token.text.lower() in interrogative_words:
            found_interrogative = True
            continue  # Skip the interrogative word itself

        if found_interrogative:
            # Check if the token should be ignored based on the ignore list and POS
            if token.text.lower() not in ignore_tokens and token.pos_ in [
                "NOUN",
                "PROPN",
                "ADJ",
            ]:
                temp_tokens.append(token.text)
            elif token.pos_ not in ["NOUN", "PROPN", "ADJ"] and temp_tokens:
                break  # End subject extraction if a non-relevant POS is encountered

    # Form the subject from collected tokens
    if temp_tokens:
        subject = " ".join(temp_tokens)

    return subject

# Filter punctuation from token text
def filter_punct(tokens):
    return [
        token
        for token in tokens
        if token.pos_ != "PUNCT" and token.text not in ["=", "-"]
    ]


def extract_subject_and_object(sentence):
    """Extracts and returns the most relevant subject and object from a given sentence, excluding stop words, with enhancements for specific patterns."""
    doc = nlp(sentence)
    subject = ""
    object_ = ""

    # List of exceptions that should not be considered stop words in this context
    stop_word_exceptions = ["local", "serviço"]

    # First, attempt to find phone numbers in the sentence
    phone_numbers = find_phone_numbers(sentence)
    if phone_numbers:
        object_ = ", ".join(phone_numbers)  # Join all found phone numbers as the object

    # Apply matcher to the doc
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if match_id == nlp.vocab.strings["EMAIL"] and not object_:
            object_ = span.text

    # Enhanced iteration through token dependencies for subject and object
    for token in doc:
        if "subj" in token.dep_ and token.pos_ in ["NOUN"]:
            subject_tokens = [
                token.text
                for token in token.subtree
                if not token.is_stop or token.text.lower() in stop_word_exceptions
            ]
            subject = " ".join(subject_tokens)
        elif "obj" in token.dep_:
            object_tokens = [token.text for token in token.subtree if not token.is_stop]
            object_ = (
                " ".join(object_tokens) if not object_ else object_
            )  # Do not override if already set by pattern matcher

    # Integration of named entities for subjects and objects not captured by dependency parsing
    if not subject:
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "NORP", "ORG", "GPE", "LOC"]:
                if not subject:
                    subject = ent.text

    # Fallback to NER and ROOT token if no object found
    if not object_:
        for ent in doc.ents:
            if not object_ and ent.label_ in ["ORG", "PERSON", "GPE", "LOC"]:
                object_ = ent.text
        if not object_:  # If still no object, use ROOT subtree
            root = [token for token in doc if token.dep_ == "ROOT"][0]
            object_tokens = filter_punct([t for t in root.subtree])
            object_ = " ".join(token.text for token in object_tokens)

    return subject, object_

def classify_sentence(sent):
    """Classifies a given sentence into categories such as Question, Statement, Command, etc."""
    text = sent.text.strip()
    cleaned_text = re.sub(r"\s+", " ", text).lower()

    # Check if the sentence ends with a question mark
    if cleaned_text.endswith("?"):
        return "Question"
    # Check if the sentence starts with an interrogative word
    elif any(cleaned_text.startswith(word) for word in interrogative_words):
        return "Question"
    # Check if the sentence contains an interrogative word followed by a comma, indicating a question
    elif any(f"{word}," in cleaned_text for word in interrogative_words):
        return "Question"
    # Check for command based on imperative verbs (simple heuristic)
    elif any(cleaned_text.startswith(verb) for verb in command_verbs):
        return "Command"
    # Default to statement if no other conditions are met
    else:
        return "Statement"

def find_questions_and_answers(txt):
    """Finds and analyzes sentences, classifying them, and extracting subjects when applicable."""
    questions_answers = []
    for line in txt.split("\n"):  # Process each line individually
        actor, sentence = extract_actor_and_sentence(line)
        if not sentence:  # Skip empty sentences
            continue

        sentence_doc = nlp(sentence)

        sentence_type = classify_sentence(sentence_doc)
        
        # For questions, focus on interrogative words and their related noun phrases
        subject = ''
        object_ = ''
        if sentence_type == "Question":
            subject = extract_subject_question(sentence)
        elif sentence_type == "Statement":
            subject, object_ = extract_subject_and_object(sentence)

        # sentiment = sentiment_analysis(sentence)
        distilbert_result, roberta_result = sentiment_analysis(sentence)

        questions_answers.append(
            {
                "sentence": sentence,
                "actor": actor,
                "type": sentence_type,
                "subject": subject,
                "object_": object_,
                "sentiment_roberta_result": roberta_result,
                "sentiment_distilbert_result": distilbert_result,
            }
        )

    return questions_answers

def main():
    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions_answers = find_questions_and_answers(txt)

    questions_answers_df = pd.DataFrame(questions_answers)

    st.title("Conversation Analysis")

    st.write("Data Visualization:")
    st.dataframe(questions_answers_df)

    # Sentiment count visualization
    st.write("Sentiment Counts:")
    sentiment_counts = questions_answers_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Additional visualizations and analyses can be added here
    # For example, filtering based on actor or sentiment
    actor_filter = st.selectbox(
        "Select actor:", ["All"] + list(questions_answers_df["actor"].unique())
    )
    if actor_filter != "All":
        filtered_data = questions_answers_df[
            questions_answers_df["actor"] == actor_filter
        ]
        st.dataframe(filtered_data)
    else:
        st.dataframe(questions_answers_df)

if __name__ == "__main__":

    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions_answers = find_questions_and_answers(txt)

    # for qa in questions_answers:
    #     print(
    #         f"Sentence: '{qa['sentence']}'\nActor: '{qa['actor']}'\nType: '{qa['type']}'\nSubject: '{qa['subject']}'\nSentiment: '{qa['sentiment']}'\n"
    #     )

    # Após processar todas as linhas do chat e armazenar em questions_answers:
    questions_answers_df = pd.DataFrame(questions_answers)

    # Exemplo de visualização dos primeiros registros
    print(questions_answers_df)

    questions_answers_df.to_csv(
        "processed_chat_data.csv", index=False
    )

    # main()

# Open File ✅
# Extrair perguntas ✅
# Extrair respostas ✅
# Simple UI ✅
# TODO: Conversation Information
# TODO: Context/People information
# TODO: Check for common question words for more comprehensive question detection
# TODO: Classificar as perguntas
# TODO: Extrair as respostas
# TODO: Classificar as respostas
# TODO: Capturar Desvios de análise
# TODO: Normalizar perguntas e respostas
# TODO: UI - Dashboard
# TODO : Use Falcon as the LLM
