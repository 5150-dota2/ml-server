from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit


app = Flask(__name__)
io = SocketIO(app)

@app.route("/", methods=["GET", "POST"])
def log():
    if request.method == "POST":
        try:
            msg = request.get_json()
        except:
            msg = jsonify(msg="Logging failed")
        io.emit("log", msg)
        return jsonify(status="done")
    return render_template("index.html")

if __name__ == "__main__":
    io.run(app, debug=True)
