# ml-server
Purpose: The dota 2 bot will communicate with this `python` server to determine the best action to take based on input data.

## Setup instruction 

[![Instructions](https://img.youtube.com/vi/hyvSAWpCjRQ/0.jpg)](https://www.youtube.com/watch?v=hyvSAWpCjRQ)

## Run server:
Prereqs:
* Python3

Steps:
1. Clone the repo: `git clone https://github.com/5150-dota2/ml-server.git`
2. Create a python3 virtual environment: run `python3 -m venv venv`. This will create a virtualenv inside a folder named `venv`
3. Activate virtualenv: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `python main.py`
6. The server is now run at `localhost:8080`
7. Copy `request.lua` into `bots` folder where the dota 2 script files reside and use `request:Send()` to communicate with the webserver
