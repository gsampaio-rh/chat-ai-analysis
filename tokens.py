import spacy
import re

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

def read_conversation_file(file_path):
    """Reads content from a specified file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        txt = file.read()
    return txt

def extract_sentence_from_line(line):
    """Extracts the sentence from a line, removing any speaker prefix."""
    # Remove possible speaker prefixes and return the sentence
    sentence = re.sub(r"^(Entrevistador:|Pessoa:)\s*", "", line).strip()
    return sentence

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


def extract_subject(sentence):
    """Extracts and returns the subject from a given sentence more accurately, considering dependency parsing."""
    doc = nlp(sentence)

    # Attempt to find the syntactic subject (nsubj) of the sentence
    subjects = [token.text for token in doc if token.dep_ in ["nsubj", "nsubjpass"]]

    # If no clear subject is found, look for nominal subjects or compound nouns that might act as subjects
    if not subjects:
        compounds_and_nouns = [
            token.text
            for token in doc
            if token.dep_ == "compound" or token.pos_ == "NOUN"
        ]
        if compounds_and_nouns:
            subjects = compounds_and_nouns

    # Combine subjects or return the most relevant noun as the subject
    subject = " ".join(subjects) if subjects else ""

    # Consider extracting keywords or noun phrases if no subjects are identified by the above method
    if not subject:
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        subject = noun_phrases[0] if noun_phrases else ""

    return subject


def classify_sentence(sent):
    """Classifies a given sentence into categories such as Question, Statement, Command, etc."""
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
        subject = extract_subject(sentence_doc)

        questions_answers.append(
            {
                "sentence": sentence,
                "actor": actor,
                "type": sentence_type,
                "subject": subject,
            }
        )

    return questions_answers

if __name__ == "__main__":
    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions_answers = find_questions_and_answers(txt)

    for qa in questions_answers:
        print(
            f"Sentence: '{qa['sentence']}'\nActor: '{qa['actor']}'\nType: '{qa['type']}'\nSubject: '{qa['subject']}'\n"
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
