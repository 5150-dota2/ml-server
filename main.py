from japronto import Application
from json import JSONDecodeError
from random import randint

def main(req):
    if req.method == "POST":
        try:
            json = req.json
            action = randint(1, json["actionListSize"])
        except (JSONDecodeError, AttributeError) as e:
            action = 1
        return req.Response(json={"action": action})
    return req.Response("hi")


app = Application()
app.router.add_route("/", main)
app.run(reload=True, debug=True)
