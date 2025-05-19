from flask import Flask, render_template, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId

# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['user_db']  # Database name
users_collection = db['users']  # Collection for storing user data

@app.route('/view_users')
def view_users():
    try:
        # Fetch all users from the 'users' collection
        users = users_collection.find()  # This returns a cursor

        # Convert MongoDB cursor to a list of dictionaries
        users_list = []
        for user in users:
            # You need to convert ObjectId to string as it is not JSON serializable
            user['_id'] = str(user['_id'])
            users_list.append(user)

        # Render the users in a template (or return as JSON for APIs)
        return jsonify(users_list)  # This returns a JSON response with the user data
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)