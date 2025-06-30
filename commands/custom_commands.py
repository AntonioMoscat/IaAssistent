#custom commands 

import re
import asyncio
import threading
from webSocket import send_ws_message

def set_timer_from_text(command: str) -> str:
    match = re.search(r"(\d+)\s*(minuto|minuti|secondo|secondi)", command)
    if not match:
        return "⏱️ Specifica una durata per il timer (es. '5 minuti')."

    value, unit = match.groups()
    value = int(value)

    seconds = value * 60 if "minut" in unit else value

    def timer_done():
        print("\n⏰ Timer terminato!")
        asyncio.run(send_ws_message("⏰ Il timer è terminato!"))

    threading.Timer(seconds, timer_done).start()
    return f"⏱️ Timer impostato per {value} {unit}."

def apri_calendario(command: str) -> str:
    return 'https://calendar.google.com/calendar'
