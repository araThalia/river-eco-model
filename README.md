# Ökologische Modellierung eines Flusssystems

Dieses Projekt simuliert die Verbreitung einer Zielart (**Species XY**) an einem dynamischen **Alpenfluss**.
Die Habitatdynamik (suitable / rejuvenation) und Populationsprozesse (juvenil/adult/senil) werden stochastisch modelliert.
Dispersal wird als Short-Distance (SDD) und Long-Distance (LDD) abgebildet.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ausführung (JSON-Config)
```bash
python src/modell_final.py --config configs/baseline.json
```
Weitere Szenarien:
```bash
python src/modell_final.py --config configs/stepstone.json
python src/modell_final.py --config configs/verbau.json
```

## Output
- CSV-Ergebnisse (mit Zeitstempel)
- Diagrammübersicht der Simulation

## Lizenz
MIT
