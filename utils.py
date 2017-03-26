import os
import json


class DataPath:
    # base = json.loads(open("config.json").read()).get("path", "")
    base = json.loads(open("config.json").read())["path"]
