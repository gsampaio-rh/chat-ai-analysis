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
