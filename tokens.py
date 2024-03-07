import spacy
import re

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

def read_conversation_file(file_path):
    """Reads content from a specified file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        txt = file.read()
    return txt

def is_question(sent):
    """Determines if a given sentence is a question by checking for a question mark and common interrogative words."""
    text = sent.text.strip()
    cleaned_text = re.sub(r"\s+", " ", text).lower()

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

    # Check if the sentence ends with a question mark
    if cleaned_text.endswith("?"):
        return True
    # Check if the sentence starts with an interrogative word
    elif any(cleaned_text.startswith(word) for word in interrogative_words):
        return True
    # Check if the sentence contains an interrogative word followed by a comma, indicating a question
    elif any(f"{word}," in cleaned_text for word in interrogative_words):
        return True
    # Additional custom rules can be added here

    return False

def find_question(sent):
    """Finds and returns questions in the given text, excluding non-question preambles."""
    # Process the text with spacy
    question = ''
    text = sent.text.strip()
    # Clean the sentence of newline characters and extra spaces
    cleaned_text = re.sub(r"\s+", " ", text)
    # Split at colon if it exists, focusing on text after the last colon which might be a direct question
    parts = cleaned_text.rsplit(":", 1)
    potential_question = parts[-1].strip() if len(parts) > 1 else parts[0]
    # Add to questions if it ends with a question mark
    if potential_question.endswith("?"):
        question = potential_question

    return question

def extract_sentence_from_line(line):
    """Extracts the sentence from a line, removing any speaker prefix."""
    # Remove possible speaker prefixes and return the sentence
    sentence = re.sub(r"^(Entrevistador:|Pessoa:)\s*", "", line).strip()
    return sentence

def extract_subject(sentence):
    """Extracts and returns the subject from a given question more accurately."""
    # Process the question with spacy
    doc = nlp(sentence.text)

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

def find_questions_and_answers(txt):
    """Finds and pairs questions with their immediate answers."""
    doc = nlp(txt)
    questions_answers = []
    sentences = list(doc.sents)
    for sent in sentences:
        line = extract_sentence_from_line(sent.text)
        questions_answers.append(
            {
                "sentence": line,
                "question": is_question(sent),
                "subject": extract_subject(sent),
            }
        )

    return questions_answers

if __name__ == "__main__":
    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions_answers = find_questions_and_answers(txt)

    for qa in questions_answers:
        print(
            f"Sentence: '{qa['sentence']}'\nSubject: '{qa['subject']}'\n"
        )

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
