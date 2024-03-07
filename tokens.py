import spacy
import re

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

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


def extract_subject(sentence, sentence_type):
    """Extracts and returns the most relevant subject from a given sentence, considering if it's a question or a statement."""

    # For questions, focus on interrogative words and their related noun phrases
    if sentence_type == "Question":
        return extract_subject_question(sentence)
    elif sentence_type == "Statement":
        return extract_subject_statement(sentence)


def extract_subject_question(sentence):
    """Defines question words and words to ignore right after interrogatives."""
    doc = nlp(sentence)

    # Use a list to temporarily store tokens after encountering a question word
    interrogative_tokens = [
        "quem",
        "qual",
        "quais",
        "onde",
        "como",
        "por que",
        "quando",
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

    subject = ""
    found_interrogative = False

    # Use a list to temporarily store tokens after encountering a question word
    temp_tokens = []

    for token in doc:
        if token.lower_ in interrogative_tokens:
            found_interrogative = True
            continue  # Jump to the next token after finding an interrogative

        if found_interrogative:
            # If the current token should be ignored, continue without adding to the subject
            if token.lower_ not in ignore_tokens and token.pos_ in [
                "NOUN",
                "PROPN",
                "ADJ",
            ]:
                temp_tokens.append(token.text)
            elif token.pos_ not in ["NOUN", "PROPN", "ADJ"] and temp_tokens:
                # If we reach a token that is not a noun, propnoun or adjective, and we have already collected tokens, for the search
                break

    # Junta os tokens temporários para formar o sujeito
    if temp_tokens:
        subject = " ".join(temp_tokens)

    return subject

def extract_subject_statement(sentence):
    """Extracts and returns the most relevant subject from a given sentence."""
    doc = nlp(sentence)

    # Attempt to use named entities first as they can be more specific
    named_entities = [
        ent.text
        for ent in doc.ents
        if ent.label_ in ["PERSON", "NORP", "ORG", "GPE", "LOC"]
    ]
    if named_entities:
        return ", ".join(
            named_entities
        )  # Return named entities as a comma-separated string if available

    # If no suitable named entities, look for noun chunks that aren't pronouns
    noun_phrases = [
        chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ != "PRON"
    ]
    if noun_phrases:
        return ", ".join(
            noun_phrases
        )  # Return noun phrases as a comma-separated string

    # As a fallback, identify standalone nouns or proper nouns directly
    nouns = [
        token.text
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and token.dep_ not in ["attr", "dobj"]
    ]
    if nouns:
        return ", ".join(nouns)

    return ""  # Return an empty string if no subject is identified


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
        subject = extract_subject(sentence_doc, sentence_type)

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
