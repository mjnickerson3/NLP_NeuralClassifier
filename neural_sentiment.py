"""
Neural Network Sentiment Classifier - Starter Template

Name: Mandi Schmuhl
Date: 10/04/2026

Instructions:
- Complete each function where indicated with TODO comments.
- Do NOT delete function definitions.
- You may add helper functions if needed.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Sample dataset
# -----------------------------
data = [
    ("I love this movie", 1),
    ("This is amazing", 1),
    ("I hate this", 0),
    ("This is terrible", 0),
    ("I really enjoyed this", 1),
    ("This was awful", 0),
]

# 1 = positive
# 0 = negative


# -----------------------------
# Text preprocessing
# -----------------------------
def tokenize(text):
    """
    Convert text to lowercase and split into tokens.
    """
    tokens = text.lower().split()
    return tokens


def build_vocabulary(dataset):
    """
    Build a vocabulary dictionary from the dataset.
    Reserve:
    0 = <PAD>
    1 = <UNK>
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for text, label in dataset:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def text_to_indices(text, vocab):
    """
    Convert text into a list of token indices.
    Unknown words should use <UNK>.
    """
    indices = []

    for token in tokenize(text):
        indices.append(vocab.get(token, vocab["<UNK>"]))

    return indices


def pad_sequence(indices, max_length, pad_value=0):
    """
    Pad or truncate a sequence to max_length.
    """
    # Pad with pad_value if sequence is too short
    # Truncate if sequence is too long
    if len(indices) < max_length:
        indices += [pad_value] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices


# -----------------------------
# Prepare dataset
# -----------------------------
def prepare_data(dataset, vocab, max_length):
    """
    Convert dataset into tensors for model training.
    """
    X = []
    y = []

    for text, label in dataset:
        # Convert text to indices
        indices = text_to_indices(text, vocab)

        # Pad the sequence
        padded = pad_sequence(indices, max_length)


        X.append(padded)
        y.append(label)

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


# -----------------------------
# Model definition
# -----------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        #Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Create hidden layer
        self.hidden = nn.Linear(embedding_dim, hidden_dim)

        # Create activation function
        self.relu = nn.ReLU()

        # Create output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, sequence_length]
        """
        # Pass through embedding layer
        embedded = self.embedding(x)

        # Mean-pool over sequence dimension
        pooled = embedded.mean(dim=1)

        # Pass through hidden layer
        hidden = self.hidden(pooled)

        #Apply ReLU
        activated = self.relu(hidden)

        # Pass through output layer
        logits = self.output(activated)

        return logits


# -----------------------------
# Prediction helper
# -----------------------------
def predict_sentiment(text, model, vocab, max_length):
    """
    Predict sentiment for a single sentence.
    """
    model.eval()

    # Convert text to indices
    indices = text_to_indices(text, vocab)

    # Pad sequence
    padded = pad_sequence(indices, max_length)

    # Convert to tensor
    input_tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        # Get logits from model
        logits = model(input_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get confidence score
        confidence = probabilities[0][predicted_class].item()

    label_name = "positive" if predicted_class == 1 else "negative"
    return label_name, confidence, probabilities


# -----------------------------
# Main program
# -----------------------------
def main():
    # Build vocabulary
    vocab = build_vocabulary(data)

    # Find max sequence length
    max_length = max(len(tokenize(text)) for text, _ in data)

    # Prepare tensors
    X_tensor, y_tensor = prepare_data(data, vocab, max_length)

    print("Vocabulary:")
    print(vocab)

    print("\nEncoded Inputs:")
    print(X_tensor)

    print("\nLabels:")
    print(y_tensor)

    # Hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 10
    hidden_dim = 8
    output_dim = 2
    learning_rate = 0.01
    epochs = 50

    # Create model
    model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    print(model)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nTraining...")
    for epoch in range(epochs):
        model.train()

        # Zero gradients
        optimizer.zero_grad()

        # Run forward pass
        logits = model(X_tensor)

        # Compute loss
        loss = criterion(logits, y_tensor)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()


        if (epoch + 1) % 10 == 0:
            # Compute predictions
            predictions = torch.argmax(logits, dim=1)

            # Compute accuracy
            accuracy = (predictions == y_tensor).float().mean().item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")

    # Test predictions
    test_sentences = [
        "I love this",
        "This is bad",
        "I do not like this",
        "This was amazing",
        "This was terrible"
    ]

    print("\nPredictions:")
    for sentence in test_sentences:
        label, confidence, probabilities = predict_sentiment(sentence, model, vocab, max_length)
        print(f"Text: {sentence}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probabilities.tolist()}")
        print("-" * 50)


if __name__ == "__main__":
    main()