import os
import json
from datetime import datetime

# Get the current directory where this file is located
current_directory = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(current_directory, "query_logs.json")

def log_query(coin, user_query, sentiment, top_results):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "coin": coin,
        "query": user_query,
        "sentiment": sentiment,
        "answers": [
            {
                "text": sentence,
                "score": round(score, 4)
            } for sentence, score in top_results
        ]
    }

    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        else:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=4)
    except Exception as e:
        print(f"Error logging query: {e}")
