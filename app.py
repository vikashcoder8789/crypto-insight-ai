@app.route('/query-check', methods=['POST'])
def check_query():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if not query_isvalid(query):
        return jsonify({"valid": False, "message": "Query not relevant"}), 200

    return jsonify({"valid": True, "message": "Query is relevant"}), 200
