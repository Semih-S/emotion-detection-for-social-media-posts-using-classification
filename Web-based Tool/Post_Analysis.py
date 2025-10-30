from flask import Flask, render_template, request, redirect, url_for
import asyncio
from atproto import AsyncClient, SessionEvent, Session
from urllib.parse import urlparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import io
import base64

# Force TensorFlow to use CPU for faster processing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Download required NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

# Create a set of stopwords for cleaning (same as used during training)
stop_words = set(stopwords.words('english'))

# Mapping from classifier number to emotion (same as used during training)
classifier_num_to_emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load the trained model
model = load_model('RNN_Tweet_Classifier.h5')

# Load the tokenizer from the saved JSON file (same as used during training)
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Set the max sequence length (same as used during training)
MAX_SEQUENCE_LENGTH = 100

def clean_text(text):
    """
    Input: Unprocessed text
    Process: Remove stopwords, Convert Emoji to useful text, Remove Hastags "#" but leave the useful text part
    Output: Cleaned text
    """
    text = emoji.demojize(text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub(r':([^:]+):', lambda m: " " + m.group(1).replace("_", " ") + " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return " ".join(filtered_tokens)

def classify_text(text):
    """
    Input: Unclassified processed text
    Process: Labelling via trained RNN model
    Output: Label which is also the predicted emotion
    """
    # Clean the text exactly as done during training
    cleaned = clean_text(text)
    # Convert the cleaned text to sequences using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    # Predict probabilities from the model
    prediction = model.predict(padded_sequence)
    class_index = np.argmax(prediction, axis=1)[0]
    predicted_emotion = classifier_num_to_emotion.get(class_index, "Unknown")
    return predicted_emotion

def url_to_at_uri(url):
    """
    Input: A bluesky URL of a post
    Process: Converting into URI
    Output: A blueksy URI of a post
    """
    parsed = urlparse(url)
    segments = parsed.path.strip('/').split('/')
    post_id = segments[-1]
    user_tag = segments[-3]
    return f"at://{user_tag}/app.bsky.feed.post/{post_id}"

async def fetch_post_thread(link):
    """
    Input: A bluesky URL of a post
    Process: Extracting most popular 10 comments/replies via url_to_at_uri function
    Output: Most popular 10 comments/replies of the input URL post
    """
    uri = url_to_at_uri(link)
    print("Converted URI:", uri)
    client = AsyncClient("https://bsky.social")

    @client.on_session_change
    async def on_session_change(event: SessionEvent, session: Session):
        print("Session event:", event, session)

    # Log in using your API credentials
    await client.login("semih-cardiff.bsky.social", "hbd4-jnvb-iaml-fh77")
    res = await client.get_post_thread(uri=uri, depth=1)
    thread = res.thread

    # Extract text from 10 replies
    replies_text = []
    for reply in thread.replies[:10]:
        try:
            replies_text.append(reply.post.record.text)
        except AttributeError:
            replies_text.append("No text available")
    return replies_text

app = Flask(__name__)

# Home page which is also the main page
@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    submitted_link = None
    replies = None
    invalid_link = False
    emotion_plot = None        # For the original emotion frequency plot
    pos_neg_plot = None        # For the final positive vs negative sentiment plot

    if request.method == 'POST':
        submitted_link = request.form.get('bluesky-link')
        # Simple check to ensure it looks like a Bluesky post link
        if "bsky.app/profile" not in submitted_link or "/post/" not in submitted_link:
            invalid_link = True
        else:
            try:
                replies = asyncio.run(fetch_post_thread(submitted_link))
            except Exception as e:
                print("Error fetching post thread:", e)
                replies = [f"Error: {e}"]

    # If we got replies and the link is valid, classify each reply
    reply_results = None
    if replies and not invalid_link:
        reply_results = []
        for reply in replies:
            emotion = classify_text(reply)
            reply_results.append((reply, emotion))
    
    # If we have classified replies, build an emotion frequency table and plot it
    if reply_results:
        # Calculate frequency of each emotion
        emotion_freq = {}
        for _, emotion in reply_results:
            emotion_freq[emotion] = emotion_freq.get(emotion, 0) + 1
        
        # Create a bar chart using matplotlib for all emotions
        plt.figure(figsize=(6, 4))
        plt.bar(emotion_freq.keys(), emotion_freq.values(), color='skyblue')
        plt.xlabel("Emotion")
        plt.ylabel("Frequency")
        plt.title("Emotion Frequency in Replies")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        emotion_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Now, create a final plot showing Positive vs Negative sentiment.
        # Define which emotions are considered positive and negative:
        positive_emotions = {"joy", "love", "suprise"}
        negative_emotions = {"sadness", "anger", "fear"}
        positive_count = sum(1 for _, emotion in reply_results if emotion in positive_emotions)
        negative_count = sum(1 for _, emotion in reply_results if emotion in negative_emotions)
        
        plt.figure(figsize=(4, 4))
        plt.bar(["Positive", "Negative"], [positive_count, negative_count], color=['green', 'red'])
        plt.xlabel("Sentiment")
        plt.ylabel("Frequency")
        plt.title("Positive vs Negative Sentiment")
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        pos_neg_plot = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close()

    return render_template("home.html", 
                           submitted_link=submitted_link, 
                           replies=reply_results, 
                           invalid_link=invalid_link,
                           emotion_plot=emotion_plot,
                           pos_neg_plot=pos_neg_plot)

# About page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
