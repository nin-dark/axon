import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the exact same model Axon uses
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1 = Should be a cache hit (semantically identical intent)
# 0 = Should NOT be a cache hit (different intent, even if words are similar)
PAIRS = [
    # Positive Pairs (True Positives)
    ("show me failed transactions", "what transactions failed", 1),
    ("how much did john spend", "total amount spent by john", 1),
    ("list all clients in New York", "give me clients located in NY", 1),
    ("what is the average deal size", "mean value of our deals", 1),
    ("top 10 sales reps", "who are the best 10 salespeople", 1),
    ("which transactions were rejected", "rejected txns", 1),
    ("show all users who signed up today", "today's new user signups", 1),
    ("count of failed payments", "how many payments failed", 1),
    ("find duplicate records", "are there any duplicates", 1),
    ("get me the highest spending customer", "who spent the most money", 1),
    ("what was our revenue last quarter", "q3 revenue total", 1),
    ("what products are out of stock", "zero inventory items", 1),
    ("list employees with salary > 100k", "staff making over 100000", 1),
    ("show me errors from yesterday", "yesterday's error logs", 1),
    ("who is the manager of the IT department", "IT department lead", 1),
    
    # Negative Pairs (Adversarial False Positives)
    ("failed transactions last week", "failed transactions last month", 0),
    ("what is the average deal size", "what is the largest deal size", 0),
    ("list all clients in New York", "list all clients in New Jersey", 0),
    ("how much did john spend", "how much did jane spend", 0),
    ("top 10 sales reps", "bottom 10 sales reps", 0),
    ("show all users who signed up today", "show all users who signed up yesterday", 0),
    ("which transactions were rejected", "which transactions were approved", 0),
    ("count of failed payments", "list of failed payments", 0),
    ("get me the highest spending customer", "get me the lowest spending customer", 0),
    ("what was our revenue last quarter", "what was our profit last quarter", 0),
    ("what products are out of stock", "what products are in stock", 0),
    ("list employees with salary > 100k", "list employees with salary < 100k", 0),
    ("show me errors from yesterday", "show me warnings from yesterday", 0),
    ("who is the manager of the IT department", "who is the manager of the HR department", 0),
    ("find duplicate records", "delete duplicate records", 0),
]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Embedding 30 query pairs...")
similarities = []
for q1, q2, label in PAIRS:
    emb1 = model.encode(q1)
    emb2 = model.encode(q2)
    sim = cosine_similarity(emb1, emb2)
    similarities.append((q1, q2, sim, label))

thresholds = np.arange(0.70, 0.96, 0.01)
best_f1 = 0
best_threshold = 0.85
best_stats = None

print(f"\n{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 50)

for t in thresholds:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for q1, q2, sim, label in similarities:
        predicted = 1 if sim >= t else 0
        if predicted == 1 and label == 1: tp += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 1: fn += 1
        else: tn += 1
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{t:.2f}       | {precision:.4f}   | {recall:.4f}   | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
        best_stats = (tp, fp, tn, fn)

print("\n" + "=" * 50)
print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"Max F1 Score:      {best_f1:.4f}")
print(f"True Positives:    {best_stats[0]}")
print(f"False Positives:   {best_stats[1]} (Cache collisions! Danger!)")
print(f"True Negatives:    {best_stats[2]}")
print(f"False Negatives:   {best_stats[3]}")
print("=" * 50)

# Update config.yaml programmatically
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(config_path, "r") as f:
    content = f.read()

new_content = re.sub(
    r"semantic_similarity_threshold:\s+[0-9.]+",
    f"semantic_similarity_threshold: {best_threshold:.2f}",
    content
)

if content != new_content:
    with open(config_path, "w") as f:
        f.write(new_content)
    print(f"\n[SUCCESS] Updated config.yaml semantic_similarity_threshold to {best_threshold:.2f}")
else:
    print("\n[INFO] config.yaml already has the optimal threshold.")
