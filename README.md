SMS Spam Detection using Word2Vec and Naive Bayes

This project demonstrates the use of Word2Vec embeddings and Gaussian Naive Bayes for SMS spam detection. It preprocesses text data, trains a Word2Vec model to generate embeddings, and utilizes a Gaussian Naive Bayes classifier for prediction.

Steps Involved
Load and Inspect the Dataset

Load the SMS Spam Collection dataset.
Display the dataset structure and first few rows.
Preprocess the Data

Tokenize text data using NLTK.
Convert text to lowercase and tokenize.
Create Word2Vec Model

Train a Word2Vec model on the tokenized text data.
Prepare Data for Naive Bayes Model

Convert Word2Vec embeddings into a format suitable for Naive Bayes.
Encode labels and split data into training and testing sets.
Train the Gaussian Naive Bayes Model

Implement a Gaussian Naive Bayes classifier using scikit-learn.
Evaluate the Model

Measure accuracy, generate confusion matrix, and create a classification report.
Function to Predict Spam or Ham

Implement a function to preprocess text, obtain Word2Vec embeddings, and predict spam or ham using the trained model.

Example Usage
python code
input_text = "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate). T&C's apply 08452810075over18's"
print("Prediction:", predict_spam_or_ham(input_text))  # Output: Prediction: ['spam']

input_text = "I'll call you later."
print("Prediction:", predict_spam_or_ham(input_text))  # Output: Prediction: ['ham']

Dependencies
pandas
nltk
gensim
scikit-learn

Usage
Clone the repository.
Install dependencies using pip install -r requirements.txt.
Run the notebook or Python script to train and evaluate the model.
