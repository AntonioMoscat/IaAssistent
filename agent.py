import importlib

TASKS = {
    "ripeti": "echo",
    "meteo": "weather_fake",
}

def dispatch(command: str) -> str:
    for keyword, module in TASKS.items():
        if keyword in command.lower():
            mod = importlib.import_module(f"tasks.{module}")
            return mod.run(command)
    return None
