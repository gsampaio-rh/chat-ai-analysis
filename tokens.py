import spacy
import re

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

def read_conversation_file(file_path):
    """Reads content from a specified file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        txt = file.read()
    return txt

def find_questions(txt):
    """Finds and returns questions in the given text, excluding non-question preambles."""
    # Process the text with spacy
    doc = nlp(txt)

    questions = []
    for sent in doc.sents:
        text = sent.text.strip()
        # Clean the sentence of newline characters and extra spaces
        cleaned_text = re.sub(r"\s+", " ", text)
        # Split at colon if it exists, focusing on text after the last colon which might be a direct question
        parts = cleaned_text.rsplit(":", 1)
        potential_question = parts[-1].strip() if len(parts) > 1 else parts[0]
        # Add to questions if it ends with a question mark
        if potential_question.endswith("?"):
            questions.append(potential_question)

    return questions


def extract_subject(question):
    """Extracts and returns the subject from a given question more accurately."""
    # Process the question with spacy
    doc = nlp(question)

    # Attempt to refine subject extraction by checking for interrogatives
    for token in doc:
        # If the token is an interrogative pronoun or determiner, get the next noun
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            for child in token.children:
                if (
                    child.dep_ in ["nsubj", "nsubjpass", "dobj", "attr", "pobj"]
                    and child.pos_ == "NOUN"
                ):
                    return child.text
        elif token.tag_ in ["WP", "WDT"]:
            for child in token.head.children:
                if (
                    child.dep_ in ["nsubj", "nsubjpass", "dobj", "attr", "pobj"]
                    and child.pos_ == "NOUN"
                ):
                    return child.text

    # Fallback to the first noun in the sentence if no clear subject is found
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text

    return ""

def create_questions_subjects_dict(txt):
    """Creates a dictionary mapping questions to their subjects."""

    questions = find_questions(txt)

    questions_subjects = {}
    for question in questions:
        subject = extract_subject(question)
        questions_subjects[question] = subject
    return questions_subjects

if __name__ == "__main__":
    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions_subjects_dict = create_questions_subjects_dict(txt)

    for question, subject in questions_subjects_dict.items():
        print(f"Question: '{question}'\nSubject: '{subject}'\n")

    # Open File ✅

    # Conversation Information
    # Context/People information

    # Extrair perguntas ✅
    # TODO: Check for common question words for more comprehensive question detection
    # Classificar as perguntas
    # Extrair as respostas
    # Classificar as respostas
    # Desvio
    # print(info)
