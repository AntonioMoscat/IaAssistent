#custom commands 

def set_timer_from_text(command: str) -> str:
    import re
    match = re.search(r"(\d+)\s*(minuto|minuti|secondo|secondi)", command)
    if not match:
        return "⏱️ Specifica una durata per il timer (es. '5 minuti')."

    value, unit = match.groups()
    value = int(value)

    seconds = value * 60 if "minut" in unit else value

    import threading

    def timer_done():
        print("\n⏰ Timer terminato!")

    threading.Timer(seconds, timer_done).start()
    return f"⏱️ Timer impostato per {value} {unit}."


def apri_calendario(command: str) -> str:
    return "📅 (Placeholder) Calendario aperto – funzione non ancora implementata."
