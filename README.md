# Conversation Analysis Tool

This Python tool processes conversation text files to extract and analyze sentences, specifically focusing on identifying questions, extracting subjects, and preparing for answer extraction. It uses the SpaCy library with the Portuguese language model to perform natural language processing tasks.

## Features

- **Question Identification**: Determines whether a sentence is a question based on punctuation and common interrogative words in Portuguese.
- **Subject Extraction**: Identifies and extracts the main subject of each question to understand the focus of the inquiry.
- **Sentence Extraction**: Removes speaker identifiers from lines of dialogue, isolating the actual conversational content.

## Requirements

- Python 3.6+
- SpaCy 2.3+
- Portuguese language model for SpaCy (`pt_core_news_sm`)

## Installation

1. Ensure Python 3.6+ is installed on your system.
2. Install SpaCy:

```bash
pip install spacy
```

3. Download and install the Portuguese language model:
```bash
python -m spacy download pt_core_news_sm
```

## Usage

1. Place your conversation text files in a directory accessible to the script.
2. Update the file_path variable in the main block of the script to point to your conversation file.
3. Run the script:

```bash
python conversation_analysis.py
```

The script will print out sentences from the conversation, identifying questions, their subjects, and preparing structures for future features such as answer extraction.

## Functions

- read_conversation_file(file_path): Reads and returns the content of a specified conversation file.
- is_question(sent): Determines if a given sentence is a question.
- find_question(sent): Extracts the question part from a sentence.
- extract_sentence_from_line(line): Isolates the sentence from a dialogue line by removing speaker prefixes.
- extract_subject(sentence): Identifies and returns the subject of a sentence.
- find_questions_and_answers(txt): Analyzes the conversation text to pair questions with their answers and extract subjects.

## Contributing

Contributions to enhance functionality, such as implementing answer extraction and refining subject identification, are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is open source and available under the MIT License.