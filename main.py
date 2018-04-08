from japronto import Application
from json import JSONDecodeError
from brain import DeepQNet


ckpt_path = "ckpt 2018-03-29 04.16.16.brain" # Path to checkpoint
net = DeepQNet(19, 13, 0.9, ckpt_path)

def main(req):
    if req.method == "POST":
        try:
            json = req.json

            # Parsing game state
            signal = (
                json["team"],
                json["isAlive"],
                json["heroX"],
                json["heroY"],
                json["heroVelX"],
                json["heroVelY"],
                json["heroFacing"],
                json["heroAnimation"],
                json["enemyIsAlive"],
                json["enemyheroX"],
                json["enemyheroY"],
                json["enemyheroVelX"],
                json["enemyheroVelY"],
                json["enemyheroFacing"],
                json["enemyheroAnimation"],
                json["creephealth"],
                json["health"],
                json["enemyhealth"],
            )

            # Reward of choosing previous action
            reward = json["reward"]

            # Update the Deep Q Net with new state and the reward transitioning from
            # the previou state to this new state
            action = net.update(reward, signal) + 1

            # Printing out some stats
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
