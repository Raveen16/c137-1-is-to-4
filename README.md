Chatbot Dataset Creation using Intent Matching Algorithm
The process of creating a dataset for the chatbot is done in two stages:

First, define the training data set file. This is done using the intents.json file.

Second, preprocess the json file. The preprocessing is done in a Python file named data_preprocessing.py

We need keywords and the respective intents to create a chatbot using an Intent matching algorithm. For example, if a user enters "I like to go for a walk", the chatbot should reply with "You can visit a park today". Here, "like" and "walk" can be the keywords for which the chatbot gives a reply to go to the park.

In intents.json, a dictionary is created with keys and values. "Tags" represent the category for the desired question, "patterns" are the expected inputs or questions from users, and "responses" are the list of corresponding responses or answers for the questions falling under that particular tag. This will act as our training dataset.

json
Copy code
{"intents": [
    {"tag": "greeting",
     "patterns": ["Hi there!!", "How are you?", "Is anyone there?","Hey","Hola", "Hello"],
     "responses": ["Hello!!, thanks for asking.", "Good to see you again.","Hi there, how can I help?"]
    },
    
    {"tag": "goodbye",
     "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
     "responses": ["See you!", "Have a nice day", "Bye! Come back again soon.","Good day"]             
    }
  ]
}
The next step is to create a Python file with the name data_preprocessing. This file is responsible for performing stemming operations and creating Bag of Words.

Let's create a python file for preprocessing of this text data. This Python file and JSON file should be in the same folder.

Create a data_preprocessing file. Let's import all the libraries required to preprocess our data.
python
Copy code
# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
Start by importing nltk. NLTK is also known as the Natural Language Toolkit. It is the Python library used for preprocessing text data. It has methods for cleaning the data and removing repetitive words. Next, from NLTK, we’ll import the PorterStemmer() class. This class is responsible to give the stem words for given words. Stemming is the process of reducing words to their root forms. It maps a group of words to the same stem even if the stem itself is not a valid word. Similarly, all the words in a sentence are reduced to their root words, and then the mapping of the sentence is done.

Import PorterStemmer from nltk.stem.

Use variable stemmer for defining an object for the PorterStemmer() class.

python
Copy code
# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
