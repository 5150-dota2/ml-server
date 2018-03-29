from japronto import Application
from json import JSONDecodeError
from brain import DeepQNet


ckpt_path = "ckpt 2018-03-29 04.16.16.brain"
net = DeepQNet(9, 8, 0.9, ckpt_path)

def main(req):
    if req.method == "POST":
        try:
            json = req.json
            signal = (
                json["team"],
                json["isAlive"],
                json["heroX"],
                json["heroY"],
                json["heroVelX"],
                json["heroVelY"],
                json["heroFacing"],
                json["heroAnimation"],
                json["health"],
            )
            reward = json["reward"]
            action = net.update(reward, signal) + 1 # Because lua index starts at 1
            print(action, reward)
        except (JSONDecodeError, AttributeError) as e:
            print("Error")
            action = 1
        return req.Response(json={"action": action})
    return req.Response("hi")


def save(req):
    net.save()
    return req.Response("Saved")


app = Application()
app.router.add_route("/", main)
app.router.add_route("/save", save)
app.run(reload=True, debug=True)
