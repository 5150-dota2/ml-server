from japronto import Application
from json import JSONDecodeError
from brain import DeepQNet


ckpt_path = "final.brain" # Path to checkpoint
net = DeepQNet(num_features=15, num_actions=9, gamma=0.9, path=ckpt_path)

def main(req):
    if req.method == "POST":
        try:
            json = req.json

            # Parsing game state
            signal = (
                json["heroX"],
                json["heroY"],
                json["heroVelX"],
                json["heroVelY"],
                json["heroFacing"],
                json["heroAnimation"],
                json["heroAADmg"],
                json["heroAttackSpeed"],
                json["numberOfNearbyEnemyHeroes"],
                json["numberOfNearbyEnemyTowers"],
                json["numberOfNearbyAlliedTowers"],
                json["numberOfNearbyEnemyCreeps"],
                json["numberOfNearbyAlliedCreeps"],
                json["numberOfNearbyNeutralCreeps"],
                json["lowestEnemyCreepHealth"],
            )

            # Reward of choosing previous action
            reward = json["reward"]

            # Update the Deep Q Net with new state and the reward transitioning from
            # the previou state to this new state
            # Note: + 1 because Lua index starts at 1
            action = net.update(json["team"], reward, signal) + 1

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
