import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Models ---
try:
    sentiment_model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    dataset = joblib.load('movies_data.pkl')
    similarity = joblib.load('recommender_model.pkl')
except Exception as e:
    messagebox.showerror("Error", f"Failed to load models: {e}")
    exit()

# --- Predict Sentiment ---
def predict_sentiment(review):
    vec = vectorizer.transform([review])
    return sentiment_model.predict(vec)[0]  # 1 = Positive, 0 = Negative

# --- Get Similar Movies ---
def get_similar_movies(movie_title):
    matches = difflib.get_close_matches(movie_title, dataset['names'].tolist(), n=1, cutoff=0.5)
    if not matches:
        return []

    close_match = matches[0]
    idx = dataset[dataset['names'] == close_match].index[0]

    sim_scores = list(enumerate(similarity[idx]))
    sorted_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]

    recommendations = [dataset.iloc[i[0]]['names'] for i in sorted_similar]
    return recommendations

# --- Handle User Input and Show Results ---
def analyze_and_recommend():
    movie_name = entry_movie.get().strip()
    review = text_review.get("1.0", tk.END).strip()

    if not movie_name or not review:
        messagebox.showwarning("Input Error", "Please enter both a movie name and a review.")
        return

    sentiment = predict_sentiment(review)

    if sentiment == 1:
        msg = f"The review is positive!\n\nYou liked '{movie_name}', so you might enjoy these similar movies:"
    else:
        msg = f"The review is negative.\n\nYou didn't like '{movie_name}'. Still, here are some similar movies you could try:"

    recommendations = get_similar_movies(movie_name)

    if not recommendations:
        messagebox.showinfo("No Match", "No matching movie found. Try another title.")
        return

    msg += "\n\n• " + "\n• ".join(recommendations)
    messagebox.showinfo("Recommendation Result", msg)

# --- GUI Setup ---
root = tk.Tk()
root.title("Movie Review Analyzer & Recommender")
root.geometry("500x400")
root.resizable(False, False)

# Labels and Inputs
tk.Label(root, text="Enter Movie Name:", font=("Arial", 12)).pack(pady=5)
entry_movie = tk.Entry(root, width=40, font=("Arial", 12))
entry_movie.pack(pady=5)

tk.Label(root, text="Write Your Review:", font=("Arial", 12)).pack(pady=5)
text_review = tk.Text(root, height=8, width=50, wrap=tk.WORD)
text_review.pack(pady=5)

# Analyze Button
tk.Button(root, text="Analyze Review & Get Recommendations", command=analyze_and_recommend, bg="blue", fg="white", font=("Arial", 12)).pack(pady=10)

# Start GUI
root.mainloop()