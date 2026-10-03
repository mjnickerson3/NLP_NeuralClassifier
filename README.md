# NLP_NeuralClassifier

##Overview
 
This project is a sentiment classifier built with PyTorch. The model uses a neural network to predict whether a sentence is positive or negative.

## Model Architexture

Embedding -> Mean Pooling -> Linear -> ReLU -> Linear -> Output

###Architecture Screenshot

<img width="532" height="154" alt="image" src="https://github.com/user-attachments/assets/596b7389-64d9-4b86-9d06-625e2a016435" />
 
##Training
The model was trained on a small dataset of positive and negative movie review sentences using CrossEntropyLoss and the Adam optimizer.

<img width="414" height="148" alt="image" src="https://github.com/user-attachments/assets/8104a63a-3463-46bf-96ed-f99ac2028cda" />

 
##Example Predictions
- "I love this" -> Positive
- "This is bad" -> Negative
- "I do not like this" -> Negative

<img width="581" height="577" alt="image" src="https://github.com/user-attachments/assets/4ae23aed-500c-4113-be38-4b9512186b29" />

##How My Model Works

My sentiment classifier starts out by changing each sentence to lowercase and splitting it into individual words. Each word is then turned into a number based on the vocabulary created from the training data. The embedding layer takes those numbers and turns them into vectors that represent the words. Since the model uses mean pooling, it averages the word vectors together to create a single representation of the sentence. That sentence representation is then passed through a hidden layer where the model learns patterns that can help determine whether the sentiment is positive or negative. The model uses a ReLU activation function, which helps it to learn more complex patterns in the data. The final output layer makes a prediction about whether the sentiment of the sentence is positive or negative. During training, the model adjusts its weights based on prediction errors so it can improve its accuracy.

##Reflection

The embedding layer turns words into vectors that the neural network can understand. It helps the model recognize that some words are related in meaning instead of treating every word as completely separate. The hidden layer is where the model starts learning patterns from the data.  For example, words like "love," "great," and "amazing" are usually linked to positive sentiment, while words like "hate," "terrible," and "awful" are associated with negative sentiment.
This model can capture patterns that logistic regression may miss because it learns its own features through the embedding and hidden layers. One thing I learned from the reading is that neural networks can learn useful information from the input data on their own. Because of this, the model can pick up on more complex patterns that a simpler model like logistic regression might not recognize.

