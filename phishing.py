import torch
import pandas as pd
import re
import os
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.serialization import add_safe_globals

# Add TfidfVectorizer to safe globals for torch loading
add_safe_globals([TfidfVectorizer])


# Define the neural network model (must match the architecture of your saved model)
class PhishingDetector(nn.Module):
    def __init__(self, numerical_features, text_features, hidden_size=128, dropout_rate=0.3):
        super(PhishingDetector, self).__init__()

        # Neural network for numerical features
        self.numerical_nn = nn.Sequential(
            nn.Linear(numerical_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Neural network for text features
        self.text_nn = nn.Sequential(
            nn.Linear(text_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Combine both networks
        self.combined_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2)  # Binary classification: phishing or legitimate
        )

    def forward(self, numerical_x, text_x):
        numerical_features = self.numerical_nn(numerical_x)
        text_features = self.text_nn(text_x)
        combined = torch.cat((numerical_features, text_features), dim=1)
        output = self.combined_nn(combined)
        return output


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what'}
    tokens = [word for word in tokens if word not in common_stopwords]
    tokens = [word if len(word) <= 5 else word[:5] for word in tokens]
    return " ".join(tokens)


# Function to analyze an email
def analyze_email(email_text, model_path='models/phishing_detector.pth', preproc_path='models/preprocessing.pth'):
    try:
        # Load preprocessing components with weights_only=False
        preproc = torch.load(preproc_path, weights_only=False)
        vectorizer = preproc['vectorizer']
        scaler = preproc['scaler']
        label_encoder = preproc['label_encoder']

        # Extract features
        processed_text = preprocess_text(email_text)

        # Extract numerical features
        features = pd.DataFrame({
            'url_count': [
                len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                               email_text))],
            'email_count': [len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_text))],
            'text_length': [len(email_text)],
            'word_count': [len(email_text.split())]
        })

        # Add suspicious keywords
        suspicious_keywords = ['urgent', 'verify', 'account', 'password', 'bank', 'click', 'confirm', 'secure',
                               'update', 'login']
        for keyword in suspicious_keywords:
            features[f'contains_{keyword}'] = [1 if keyword in email_text.lower() else 0]

        # Scale numerical features
        numerical_features = scaler.transform(features)

        # Vectorize text
        text_features = vectorizer.transform([processed_text]).toarray()

        # Create model instance (with matching dimensions)
        model = PhishingDetector(
            numerical_features=numerical_features.shape[1],
            text_features=text_features.shape[1]
        )

        # Load trained model weights - use weights_only=False for compatibility
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()

        # Make prediction
        with torch.no_grad():
            numerical_tensor = torch.FloatTensor(numerical_features)
            text_tensor = torch.FloatTensor(text_features)
            outputs = model(numerical_tensor, text_tensor)
            _, predicted = torch.max(outputs, 1)

        # Decode prediction
        result = label_encoder.inverse_transform([predicted.item()])[0]

        # Calculate confidence
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted.item()].item() * 100

        return {
            'prediction': result,
            'confidence': confidence,
            'is_phishing': bool(predicted.item())
        }

    except Exception as e:
        return {'error': str(e)}


# Add a command line interface to make it easier to use
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect phishing in emails using the trained model")
    parser.add_argument("--text", type=str, help="Email text to analyze")
    parser.add_argument("--file", type=str, help="Path to a text file containing the email")
    args = parser.parse_args()

    email_text = ""

    # Get email text from file or command line argument
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                email_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            exit(1)
    elif args.text:
        email_text = args.text
    else:
        # Use a sample email for testing if no input is provided
        email_text = """
        Dear Customer,
        We have detected suspicious activity on your account. Please verify your account information
        by clicking on the link below:
        http://secureminty.phishing-example.com/verify?id=12345
        If you do not verify within 24 hours, your account will be locked.

        Thank you,
        Bank Security Team
        """
        print("No input provided, using sample phishing email for testing.")
    print(email_text)
    # Analyze the email
    result = analyze_email(email_text)

    if 'error' in result:
        print(f"Error analyzing email: {result['error']}")
    else:
        print(f"\nPHISHING ANALYSIS RESULTS:")
        print(f"---------------------------")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Is phishing: {'Yes' if result['is_phishing'] else 'No'}")