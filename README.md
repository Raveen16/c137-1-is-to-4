The process of creating a dataset for the chatbot is done in two stages:
● First, define the training data set file. 
    This is done using the intents.json file.
● Second, preprocessing the json file. 
    The preprocessing is done in a Python file named data_preprocessing.py


We need keywords and the respective intents to create a chatbot using an Intent matching algorithm.


For example: 
    ‘I like to go for a walk’
Chatbot reply: 
    ‘You can visit a park today’.
Here, ‘like’ and ‘walk’ can be the keywords for which chatbot gives reply to go to the park.


In intents.json a dictionary is created with keys and values.
‘tags’ represent the category for the desired question.
‘patterns’ are the expected inputs or questions from users.
‘responses’ are the list of corresponding responses or answers for the questions falling under that particular tag.


*This will act as our training dataset.*

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


The next step is to create a Python file with the name data_preprocessing . 
This file is responsible for performing Stemming operation and creating Bag of Words.


Let’s create a python file for preprocessing of this text data.
This Python file and JSON file should be in the same folder.
1. Create a data_preprocessing file. 
    Let’s import all the libraries required to preprocess our data.

2. Start by importing nltk. 
    NLTK is also known as the *Natural Language Toolkit*
    It is the Python library used for preprocessing text data. 
    It has methods for cleaning the data and removing repetitive words.
    Next, from NLTK we’ll import the PorterStemmer() class.
    This class is responsible to give the stem words for given words.

    Stemming is the process of reducing words to their root forms. 
    It maps a group of words to the same stem even if the stem itself is not a valid word.
    Similarly, all the words in a sentence are reduced to their root words and then the mapping of the sentence is done.

3. Import PorterStemmer from nltk.stem
4. Use variable stemmer for defining object for the PorterStemmer() class.

*code*
#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()



5. Next, import all the required libraries such as json for reading and processing JSON data.
6. Import pickle.
    Pickle is the Python library that converts lists, dictionaries, and other objects into streams of zero and one.
    This will be helpful to store preprocessed training data.
7. Install numpy library too as the training dataset has to be Numpy arrays

*code*
import json
import pickle
import numpy as np


8. After importing all the desired libraries, our first step is to load our data. 
    For storing tags, patterns and responses from json file create three lists such as words, classes and word_tags_list.

*code*
words=[]
classes = []
word_tags_list = []

9. Also, create a list of the symbols by name ignore_words. 
    These symbols should be avoided for processing text as only words are needed for tokenizing.
10. Create a variable train_data_file for reading the intents.json file.
11. Load this json file into variable name intents using the json.load() function

*code*
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

*Steps for Stemming:*
Define a function to get the stem word from the list of words.
Create an empty list for storing stem_words.
Write a for loop for looping through each word. 
Convert the word into lower case to maintain the uniformity between all the words. 
As the words ‘Play’ and ‘play’ are tokenized differently.
Use the stem() function to change the word into its stem or root word.
Append this root word into the list stem_words. 
This list will contain all the words converted into their stem words and the punctuation will be removed.


def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words


*Creating chatbot corpus:*
Creating a chatbot corpus that is set of words that can be expected from user as input. 
To create a corpus, we first have to extract these words from intents.json. 
We are using a for loop to iterate through each intent or tag.
Write a for loop for adding words and tags into lists.
    1. Tokenize each pattern and store it in the pattern_word variable.
    2. The extend() method will add all the tokenized patterns into the list words.
    3. Create word_tags_list, lists of words and tags in the form [‘word’, ‘tag’]. 
This is an empty list that is appended by words and tags. 
In the append these tokenized patterns and respective tags.


for intent in intents['intents']:
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))


4. Now, create a list of tags known as classes. Append all the tags to the list.
5. Also, call the get_stem_words() function to create the list of stem words while excluding ignore_words.


if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)


Check these lists now for creating word corpus.

print(stem_words)
print(word_tags_list[0]) 
print(classes) 


*Steps to check output*
1. Go to the command prompt and traverse the working folder as we did in class 110.
2. Create a virtual environment for testing the model using 
    python -m venv <name_of_the _environment> 
3. Activate the virtual environment using following command:
    <name_of_the_environment>\Scripts\activate
4. Install NLTK using command pip install nltk. Also, install all the required libraries in the environment.
5. Run the python file data_preprocessing .py and check the output




1. Define a function create_bot_corpus that takes stem_words and classes parameters for creating a word corpus.
2. List of stem_words and classes are converted into set to get unique words from these lists. 
    Again they are converted into list and stored in sorted manner.
3. Create two pickle files by name words.pkl and classes.pkl. These are then written by pickle.dump() method for creating thetraining      dataset. ‘wb’ stands for write in binary mode. Thus the data is stored in the binary form(0 and 1) in these files

4. Thus, this function returns the sorted stem_word list and sorted list of classes.
5. Call the function to create the corpus.



#Create word corpus for chatbot
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes




stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)


Now our next step is to create Bag Of Words.
A Bag Of Words is a representation of text that shows the occurrence of words in a sentence.
Two things are required for creating BOW. 
A vocabulary of known words. 
This is given by stem_words and array to represent the BOW representation of the sentence.
