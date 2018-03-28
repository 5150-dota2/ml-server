from japronto import Application
from json import JSONDecodeError
from brain import DeepQNet

net = DeepQNet(4, 9, 0.05)

def main(req):
    if req.method == "POST":
        try:
            json = req.json
            signal = (json["heroX"], json["heroY"], json["heroAnimation"], json["health"])
            reward = json["reward"]
            action = net.update(reward, signal)
            print(action, reward)
        except (JSONDecodeError, AttributeError) as e:
            action = 1
        return req.Response(json={"action": action})
    return req.Response("hi")


app = Application()
app.router.add_route("/", main)

try:
    app.run(reload=True, debug=True)
except KeyboardInterrupt as e:
    print("Exitting...")
