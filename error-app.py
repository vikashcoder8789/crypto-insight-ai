from flask import Flask, request, jsonify

app = Flask(__name__)

VALID_COINS = ["bitcoin", "ethereum", "dogecoin", "solana"]

def is_valid_query(query):
    if not query or not query.strip():
        return "Please provide both coin and query."
    if len(query.split()) < 3:
        return "Your query is too short or unclear. Please provide more context."
    if not any(char.isalpha() for char in query):
        return "Query must contain readable text. Please avoid using only symbols or numbers."
    if "crypto" not in query.lower() and not any(coin in query.lower() for coin in VALID_COINS):
        return "Query seems unrelated to crypto. Try asking something like 'What's the trend of Bitcoin this week?'"
    return None

@app.route("/ask", methods=["POST"])
def handle_query():
    data = request.get_json()
    coin = data.get("coin", "").lower()
    query = data.get("query", "")

    if not coin or not query:
        return jsonify({"error": "Please provide both coin and query."}), 400

    if coin not in VALID_COINS:
        return jsonify({"error": "Unsupported coin. Choose from the given coins."}), 400

    validation_msg = is_valid_query(query)
    if validation_msg:
        return jsonify({"error": validation_msg}), 400

    # Placeholder for NLP model
    #print
    return jsonify({"result": f"Processed your query: '{query}' for {coin.title()}"}), 200

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": "Something went wrong on our side. We're looking into it."}), 500

if __name__ == "__main__":
    app.run(debug=True)
