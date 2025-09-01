
#%%run
import argparse
import json
import math
import random
from typing import List
import numpy
import csv
from time import strftime
import matplotlib.pyplot as plt
import pandas as pd

dbug = False

def dbprint(*out):
    if dbug:
        print(out)

def event(p):
    """returns true if an event occurs with probability p (0..1)"""
    p = max(0.0, min(1.0, float(p)))
    e = numpy.random.choice([1, 0], 1, p=[p, 1 - p])[0]
    return e

def turnoverWeight(patch, limits=[(10,0.5),(20,0.0)]):
    """ calculates turnover weight 0..1
    limits: list of tuples (age, weight) sorted ascending by age
    """
    age = patch.getAge()
    weight = 1
    for l in limits:
        if age > l[0]:
            weight = l[1]
        dbprint("turnover set to",weight)
    return weight

class Population():
    sdd_dist = 20
    min_fertility_age = 3
    max_fertility_age = 70
    max_age = 100

    def __init__(self, age=0):
        self.age = max(0, min(age, self.max_age))

    def __str__(self):
        return("Population with age "+str(self.age)+"; stage "+self.getLifeHistoryStage())

    def getAge(self): return self.age
    def isFertile(self): return (self.age >= self.min_fertility_age and self.age <= self.max_fertility_age)
    def isSenile(self): return self.age >= self.max_fertility_age
    def diesOff(self): return self.age > self.max_age

    def getLifeHistoryStage(self):
        if self.age <= self.min_fertility_age: return "J"
        if self.age > self.max_fertility_age: return "S"
        return "A"

    def tick(self):
        self.age = self.age + 1

class Patch():
    """Habitat patch"""
    time_for_colonialisation = 2
    max_rejuv_age = Population.min_fertility_age + time_for_colonialisation
    max_species_age = 90

    def __init__(self, suitable=False, age=0, locked=False):
        self.age = age
        self.suitable = suitable
        self.population = None
        self.locked = locked

    def __str__(self):
        info = "Habitatpatch with age "+str(self.age)+"; "
        info += "generally suitable; " if self.suitable else "generally unsuitable; "
        if self.allowsRejuvenation():
            info += "rejuvenation possible; "
        return info

    def getAge(self): return self.age
    def ageIs(self,age): self.age = age

    def allowsRejuvenation(self):
        return self.suitable and self.age <= self.max_rejuv_age

    def allowsSpecies(self):
        return self.suitable and self.age <= self.max_species_age

    def isSuitable(self): return self.suitable
    def isPopulated(self): return self.population is not None
    def isLocked(self): return self.locked
    def lock(self): self.locked = True
    def unlock(self): self.locked = False
    def getPopulation(self): return self.population

    def populate(self,population):
        if self.allowsSpecies():
            self.population = population
            return 1 if population is not None else 0
        else:
            dbprint("Achtung: ungeeignet für Populationen")
            return 0

    def setUnsuitable(self):
        if self.locked:
            dbprint("cannot change suitability, patch is locked")
            return
        self.suitable = False
        self.population = None
        self.age = 0

    def setSuitable(self):
        if self.locked:
            return
        self.suitable = True
        self.age = 0

    def setSuitable_sg(self):
        self.unlock()
        self.suitable = True
        self.age = 0

    def isPatchAge(self):
        if self.locked:
            self.age = 3

    def tick(self):
        self.age = self.age + 1
        if self.population:
            self.population.tick()
        self.isPatchAge()
        if not self.allowsSpecies():
            self.population = None
        if self.population and self.population.diesOff():
            self.population = None

class River():
    patchlength = Population.sdd_dist

    def __init__(self, length=100):
        self.age = 0
        self.patchlist = [Patch() for _ in range(length)]

    def __str__(self):
        out = ""
        for patch in self.patchlist:
            if patch.isSuitable() and not patch.allowsRejuvenation():
                out += "o"
            elif patch.isSuitable() and patch.allowsRejuvenation():
                out += "+"
            else:
                out += "X" if patch.isLocked() else "-"
        return out

    def populationStatus(self):
        out = ""
        for patch in self.patchlist:
            pop = patch.getPopulation()
            if pop:
                out += "A" if pop.isFertile() else ("S" if pop.isSenile() else "J")
            else:
                out += " "
        return out

    def turnover(self, p_turnover, turnoverweights):
        turned_suitable = 0
        turned_unsuitable = 0
        for patch in self.patchlist:
            if event(p_turnover * turnoverWeight(patch, turnoverweights)):
                if patch.isSuitable():
                    patch.setUnsuitable()
                    turned_unsuitable += 1
                else:
                    patch.setSuitable()
                    turned_suitable += 1
        return (turned_suitable, turned_unsuitable)

    def lddispersal(self, p_ldd=0, ldd_kernel=None, bidirectional=True):
        lddevents = 0
        if not ldd_kernel: return 0
        for pos, patch in enumerate(self.patchlist):
            pop = patch.getPopulation()
            if pop and pop.isFertile() and event(p_ldd):
                for idx, p in enumerate(ldd_kernel, start=1):
                    # downstream
                    npos = pos + idx
                    if npos < len(self.patchlist):
                        neighbor = self.patchlist[npos]
                        if neighbor.allowsRejuvenation() and event(p):
                            lddevents += neighbor.populate(Population(age=1))
                    if not bidirectional:
                        continue
                    # upstream
                    npos = pos - idx
                    if npos >= 0:
                        neighbor = self.patchlist[npos]
                        if neighbor.allowsRejuvenation() and event(p):
                            lddevents += neighbor.populate(Population(age=1))
        return lddevents

    def dispersal(self, p_dispersal=0.1):
        sddevents = 0
        for pos, patch in enumerate(self.patchlist):
            pop = patch.getPopulation()
            if pop and pop.isFertile():
                up = self.patchlist[pos-1] if (pos-1) >= 0 else None
                dn = self.patchlist[pos+1] if pos+1 < len(self.patchlist) else None
                if up and up.allowsRejuvenation() and event(p_dispersal):
                    sddevents += up.populate(Population(age=1))
                if dn and dn.allowsRejuvenation() and event(p_dispersal):
                    sddevents += dn.populate(Population(age=1))
        return sddevents

    # reporting
    def getNumberPopulatedPatches(self):
        return sum(1 for p in self.patchlist if p.getPopulation())

    def getNumberFertilePopulations(self):
        return sum(1 for p in self.patchlist if (p.getPopulation() and p.getPopulation().isFertile()))

    def getNumberSuitablePatches(self):
        return sum(1 for p in self.patchlist if p.isSuitable())

    def getNumberRejuvenationPatches(self):
        return sum(1 for p in self.patchlist if p.allowsRejuvenation())

    def getAveragePopulationAge(self):
        ages = [p.getPopulation().getAge() for p in self.patchlist if p.getPopulation()]
        return (sum(ages)/len(ages)) if ages else 0

    def getAveragePatchAge(self):
        ages = [p.getAge() for p in self.patchlist if p.isSuitable()]
        return (sum(ages)/len(ages)) if ages else 0

    # events
    def flood(self, p_turnover, destructionbias=1, popdestruction=1):
        p_create = p_turnover
        p_destroy = p_turnover * destructionbias
        P_unpopulate = p_turnover * popdestruction
        for patch in self.patchlist:
            if not patch.isSuitable():
                if event(p_create):
                    patch.setSuitable()
            if patch.isSuitable():
                if event(p_destroy):
                    patch.setUnsuitable()
                elif patch.isPopulated() and event(P_unpopulate):
                    patch.populate(None)

    def verbau(self, p_from, p_to):
        for patch in self.patchlist[p_from:p_to]:
            patch.setUnsuitable()
            patch.lock()

    def verbau_points(self, indices):
        for idx in indices or []:
            if 0 <= idx < len(self.patchlist):
                self.patchlist[idx].setUnsuitable()
                self.patchlist[idx].lock()

    def stepstone(self, indices):
        for idx in indices or []:
            if 0 <= idx < len(self.patchlist):
                self.patchlist[idx].setSuitable()
                self.patchlist[idx].lock()

    def sgebiet(self, sg_from, sg_to):
        for patch in self.patchlist[sg_from:sg_to]:
            patch.unlock()
            patch.setSuitable_sg()
            patch.lock()

    def tick(self):
        self.age = self.age + 1
        for patch in self.patchlist:
            patch.tick()

    def getPatchlist(self) -> List['Patch']:
        return self.patchlist

################################## MODEL ##################################

def model(ensemblesize=100,
          river_length=100,
          timesteps=200,
          prop_suitable=0.1,
          prop_rejuvenate=0.2,
          prop_populated=0.1,
          populate_only=None,
          verbau=None,
          sg_year=None,
          sgebiet=None,
          stepstone_list=None,
          flood=None,
          max_habitat_age=60,
          p_turnover=0.05,
          turnoverweights=None,
          p_dispersal=0.5,
          p_ldd=1,
          ldd_kernel=None,
          modelname="",
          modeldescription="",
          patches_to_verbau=None,
          save=False):

    if turnoverweights is None: turnoverweights = []
    if ldd_kernel is None: ldd_kernel = [0.5,0.2,0.1,0.1]
    if populate_only is None: populate_only = []
    if verbau is None: verbau = []
    if flood is None: flood = {}

    if p_ldd:
        modelname = modelname + "LDD" + str(ldd_kernel) + "_"

    riverlist: List[River] = []

    for i in range(ensemblesize):
        r = River(river_length)

        # init patches
        for patch in r.getPatchlist():
            if event(prop_suitable):
                patch.setSuitable()
                if event(prop_rejuvenate):
                    age = random.randint(1, Patch.max_rejuv_age)
                else:
                    age = random.randint(Patch.max_rejuv_age + 1, max_habitat_age)
                patch.ageIs(age)

        # verbau & stepstones
        if verbau:
            if isinstance(verbau, list) and len(verbau) == 2:
                r.verbau(verbau[0], verbau[1])
        if patches_to_verbau:
            r.verbau_points(patches_to_verbau)
        if stepstone_list:
            r.stepstone(stepstone_list)

        # initial populations
        for patch in r.getPatchlist():
            if populate_only:
                if patch.isSuitable() and r.getPatchlist().index(patch) in populate_only:
                    if event(prop_populated):
                        popage = random.randint(1, patch.getAge())
                        patch.populate(Population(popage))
            else:
                if patch.isSuitable() and event(prop_populated):
                    popage = random.randint(1, patch.getAge())
                    patch.populate(Population(popage))

        riverlist.append(r)

    resultlist = []

    print("################## initial state")
    for r in riverlist:
        print(r.populationStatus())
        print(r)

    population_year = []

    for year in range(timesteps):
        rid = 0
        rl2 = []

        for r in riverlist:
            # flood event (dict: year -> [p_turnover, destructionbias, popdestruction])
            if flood and (year in flood.keys()):
                fp = flood[year]
                r.flood(p_turnover=fp[0], destructionbias=fp[1], popdestruction=fp[2])

            rid += 1

            result = {
                "year"                 : year,
                "river"                : rid,
                "numSuitablePatches"   : r.getNumberSuitablePatches(),
                "numRejuvPatches"      : r.getNumberRejuvenationPatches(),
                "numPopPatches"        : r.getNumberPopulatedPatches(),
                "numFertilePop"        : r.getNumberFertilePopulations(),
                "avgPatchAge"          : r.getAveragePatchAge(),
                "avgPopAge"            : r.getAveragePopulationAge(),
            }
            result["populated"] = 1 if result["numPopPatches"] > 0 else 0

            # Turnover
            p_to = p_turnover[year] if isinstance(p_turnover, list) else p_turnover
            turned_suitable, turned_unsuitable = r.turnover(p_to, turnoverweights)
            turned = turned_suitable + turned_unsuitable

            # Dispersal
            sddevents = r.dispersal(p_dispersal)
            lddevents = r.lddispersal(p_ldd, ldd_kernel)

            # optional recolonisation check beyond obstacles
            if patches_to_verbau:
                xyz = 0
                max_verbau = max(patches_to_verbau)
                for patch in r.getPatchlist():
                    if r.getPatchlist().index(patch) >= max_verbau and patch.isPopulated():
                        xyz += 1
                if xyz > 0:
                    population_year.append({'rid': rid, 'year': year})

            r.tick()

            result.update({
                "turned": turned,
                "turned_suitable": turned_suitable,
                "turned_unsuitable": turned_unsuitable,
                "sddEvents": sddevents,
                "lddEvents": lddevents
            })
            resultlist.append(result)
            rl2.append(result["numPopPatches"])

    header = list(resultlist[0].keys())

    print("################## last state")
    import pandas as pd
    df_py = pd.DataFrame(population_year)
    if not df_py.empty:
        df_py = df_py.sort_values(by=['year'])
    print(r.populationStatus())
    print(r)
    print(df_py)
    df_py.to_csv(f'population_year_{modelname}.csv', index=False)

    savename = modelname + strftime("%y%m%d_%H%M") + ".csv"
    if save:
        with open(savename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for data in resultlist:
                writer.writerow(data)
        print("Model saved as: ", savename)

    # quickplot
    try:
        df = pd.DataFrame(resultlist)
        ag = df.pivot_table(index="year", aggfunc='mean')
        ag_sd = df.pivot_table(index="year", aggfunc='std')

        fig = plt.figure(figsize=(8,10))
        fig.suptitle("Simulation output")

        p11 = fig.add_subplot(221)
        p12 = fig.add_subplot(222)
        p21 = fig.add_subplot(223)
        p22 = fig.add_subplot(224)

        p11.errorbar(ag.index, ag['numSuitablePatches'], yerr=ag_sd['numSuitablePatches'])
        p11.errorbar(ag.index, ag['numRejuvPatches'], yerr=ag_sd['numRejuvPatches'])
        p11.set_title("patches")

        p12.errorbar(ag.index, ag['numPopPatches'], yerr=ag_sd['numPopPatches'])
        p12.errorbar(ag.index, ag['numFertilePop'], yerr=ag_sd['numFertilePop'])
        p12.set_title("populations")

        p21.errorbar(ag.index, ag['avgPopAge'], yerr=ag_sd['avgPopAge'])
        p21.errorbar(ag.index, ag['avgPatchAge'], yerr=ag_sd['avgPatchAge'])
        p21.set_title("ages")

        p22.errorbar(ag.index, ag['sddEvents'], yerr=ag['sddEvents'])
        p22.errorbar(ag.index, ag['lddEvents'], yerr=ag['lddEvents'])
        p22.set_title("dispersal")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)

    return resultlist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run river eco model with JSON config")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "flood" in config and isinstance(config["flood"], dict):
        config["flood"] = {int(k): v for k, v in config["flood"].items()}

    model(**config)
