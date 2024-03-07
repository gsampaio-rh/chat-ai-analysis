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

        #TODO: Check for common question words for more comprehensive question detection

    return questions

if __name__ == "__main__":
    file_path = "sample_chat.txt"
    txt = read_conversation_file(file_path)

    questions = find_questions(txt)
    print(questions)

    # Open File
    # Conversation Information
    # Context/People information
    # Extrair perguntas
    # Classificar as perguntas
    # Extrair as respostas
    # Classificar as respostas
    # Desvio
    # print(info)
