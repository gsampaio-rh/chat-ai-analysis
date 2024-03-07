import spacy
# from spacy import displacy  # Uncomment if needed for visualization
# import re  # Uncomment if regular expressions are needed later

# Load the Portuguese language model
nlp = spacy.load("pt_core_news_sm")

def read_conversation_file(file_path):
  """Reads content from a specified file path."""
  with open(file_path, 'r', encoding='utf-8') as file:
    txt = file.read()
  return txt

def find_questions(txt):
  """Finds and returns questions in the given text."""
  # Process the text with spacy
  doc = nlp(txt)
  
  # Find sentences that end with a question mark
  questions = [sent.text for sent in doc.sents if sent.text.strip().endswith('?')]
  
  # Future enhancement: Check for common question words for more comprehensive question detection
  
  return questions

if __name__ == '__main__':
    file_path = 'sample_chat.txt'
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
    # print(info)