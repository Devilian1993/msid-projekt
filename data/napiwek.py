#!/usr/bin/env python

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── Universe of discourse ────────────────────────────────────────────────────
jakosc_posilku  = ctrl.Antecedent(np.linspace(0, 5, 500), 'jakosc_posilku')
jakosc_obslugi  = ctrl.Antecedent(np.linspace(0, 5, 500), 'jakosc_obslugi')
wysokosc_napiwku = ctrl.Consequent(np.linspace(0, 30, 500), 'wysokosc_napiwku',
                                    defuzzify_method='centroid')

# ── Membership functions (trimf) ─────────────────────────────────────────────
# Input 1 – jakosc_posilku
jakosc_posilku['slaba']  = fuzz.trimf(jakosc_posilku.universe,  [-2.5, 0.0, 2.5])
jakosc_posilku['srednia'] = fuzz.trimf(jakosc_posilku.universe, [ 0.0, 2.5, 5.0])
jakosc_posilku['dobra']  = fuzz.trimf(jakosc_posilku.universe,  [ 2.5, 5.0, 7.5])

# Input 2 – jakosc_obslugi
jakosc_obslugi['slaba']  = fuzz.trimf(jakosc_obslugi.universe,  [-2.5, 0.0, 2.5])
jakosc_obslugi['srednia'] = fuzz.trimf(jakosc_obslugi.universe, [ 0.0, 2.5, 5.0])
jakosc_obslugi['dobra']  = fuzz.trimf(jakosc_obslugi.universe,  [ 2.5, 5.0, 7.5])

# Output – wysokosc_napiwku
wysokosc_napiwku['niska']  = fuzz.trimf(wysokosc_napiwku.universe, [-15,  0, 15])
wysokosc_napiwku['srednia'] = fuzz.trimf(wysokosc_napiwku.universe, [  0, 15, 30])
wysokosc_napiwku['wysoka'] = fuzz.trimf(wysokosc_napiwku.universe, [ 15, 30, 45])

# ── Rules (AND = min, aggregation = max, implication = min) ──────────────────
# Rule matrix (posilku \ obslugi):
#   slaba   srednia  dobra
#   niska   niska    srednia   ← posilku=slaba
#   niska   srednia  wysoka    ← posilku=srednia
#   srednia wysoka   wysoka    ← posilku=dobra

rules = [
    ctrl.Rule(jakosc_posilku['slaba']  & jakosc_obslugi['slaba'],   wysokosc_napiwku['niska'],   and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['srednia'] & jakosc_obslugi['slaba'],  wysokosc_napiwku['niska'],   and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['dobra']  & jakosc_obslugi['slaba'],   wysokosc_napiwku['srednia'], and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['slaba']  & jakosc_obslugi['srednia'], wysokosc_napiwku['niska'],   and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['srednia'] & jakosc_obslugi['srednia'],wysokosc_napiwku['srednia'], and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['dobra']  & jakosc_obslugi['srednia'], wysokosc_napiwku['wysoka'],  and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['slaba']  & jakosc_obslugi['dobra'],   wysokosc_napiwku['srednia'], and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['srednia'] & jakosc_obslugi['dobra'],  wysokosc_napiwku['wysoka'],  and_func=np.fmin),
    ctrl.Rule(jakosc_posilku['dobra']  & jakosc_obslugi['dobra'],   wysokosc_napiwku['wysoka'],  and_func=np.fmin),
]

# ── Control system ───────────────────────────────────────────────────────────
napiwek_ctrl = ctrl.ControlSystem(rules)
napiwek_sim  = ctrl.ControlSystemSimulation(napiwek_ctrl)

# ── Interactive loop ─────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════╗
║        System FIS – Wysokość Napiwku             ║
║  Mamdani | AND=min | AGG=max | Defuzz=centroid   ║
╚══════════════════════════════════════════════════╝
Inputs range: [0, 5]   Output range: [0, 30]
Type 'q' at any prompt to quit.
"""

print(BANNER)

while True:
    # ── jakosc_posilku ───────────────────────────────────────────────────────
    raw = input("Jakość posiłku   [0–5]: ").strip()
    if raw.lower() == 'q':
        print("Do widzenia!")
        break
    try:
        val_posilku = float(raw)
        if not (0 <= val_posilku <= 5):
            raise ValueError
    except ValueError:
        print("  ⚠  Podaj liczbę z zakresu [0, 5].\n")
        continue

    # ── jakosc_obslugi ───────────────────────────────────────────────────────
    raw = input("Jakość obsługi   [0–5]: ").strip()
    if raw.lower() == 'q':
        print("Do widzenia!")
        break
    try:
        val_obslugi = float(raw)
        if not (0 <= val_obslugi <= 5):
            raise ValueError
    except ValueError:
        print("  ⚠  Podaj liczbę z zakresu [0, 5].\n")
        continue

    # ── Inference ────────────────────────────────────────────────────────────
    napiwek_sim.input['jakosc_posilku'] = val_posilku
    napiwek_sim.input['jakosc_obslugi'] = val_obslugi
    napiwek_sim.compute()

    result = napiwek_sim.output['wysokosc_napiwku']
    print(f"\n  ➤  Wysokość napiwku: {result:.2f} %\n")