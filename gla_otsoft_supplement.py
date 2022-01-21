import io
import pandas as pd
import re
import random
import numpy as np
from datetime import datetime


# path to input file (.txt formatted as for OTSoft GLA)
FILE = "./SE_input_GLA.txt"

# customize settings as appropriate #########
INIT_WEIGHTS = {}  # dictionary of constraint-value pairs, if necessary
INIT_F = 100
INIT_M = 100
MAGRI = True
SPECGENBIAS = 20
LEARNING_TRIALS = [50000, 50000, 50000, 50000]
EXPERIMENTNUM = "exp01"
LEARNING_R_F = [2, 0.2, 0.02, 0.002]
LEARNING_R_M = [2, 0.2, 0.02, 0.002]
LEARNING_NOISE_F = [2, 2, 2, 2]
LEARNING_NOISE_M = [2, 2, 2, 2]
# end user-customized settings ##############

# constraints
star_F = "*F"
star_O = "*õ"
IdBkSyl1 = "Id(Bk)Syl1"
IdBkRt = "Id(Bk)"
AgrBk = "Agr(Bk)"
GMH_F = "GMHF"
star_e = "*e"
GMH_e = "GMHe"
GMH_O = "GMHõ"

# other constants
m = "markedness"
f = "faithfulness"
ctype = "constraint type"
g1 = "group 1 strings"
g2 = "group 2 strings"


# this class represents a GLA learner with a particular set of attributes:
# - whether or not to use the Magri update rule
# - how big to make the a priori difference between specific and general faithfulness constraint values
# - experiment number, initial weights, noise, learning rates, and constraints as constants defined above
class Learner:

    def __init__(self, infile, magri=True, specgenbias=0):
        self.file = infile
        self.historyfile = infile.replace(".txt", "_HISTORY" + str(LEARNING_TRIALS[0]) + ".txt")
        self.resultsfile = infile.replace(".txt", "_RESULTS" + str(LEARNING_TRIALS[0]) + ".txt")
        if EXPERIMENTNUM is not None and EXPERIMENTNUM != "":
            afterlastbackslash = self.file.rfind("\\") + 1
            self.historyfile = self.historyfile[0:afterlastbackslash] + EXPERIMENTNUM + " - " + self.historyfile[afterlastbackslash:]
            self.resultsfile = self.resultsfile[0:afterlastbackslash] + EXPERIMENTNUM + " - " + self.resultsfile[afterlastbackslash:]
        self.magri = magri
        self.constraints = []
        self.weights = {}
        self.tableaux_list = []
        self.training_tableaux_list = []
        self.specgenbias = specgenbias

    def set_tableaux(self, tableaux_list):
        self.tableaux_list = tableaux_list
        # only try and learn from the tableaux that have input frequency information
        self.training_tableaux_list = [t for t in tableaux_list if sum(t["frequency"].values) > 0]

    # read input from a file formatted as for OTSoft GLA learner
    def read_input(self):
        with io.open(self.file, "r") as infile:
            df = pd.read_csv(infile, sep="\t", header=1, keep_default_na=False)
            df.rename(
                columns=({'Unnamed: 0': 'input', 'Unnamed: 1': 'candidate', 'Unnamed: 2': 'frequency'}),
                inplace=True,
            )

            rules = {}
            for colname in df.columns[3:]:
                contents = list(df[colname].values)

                if contents[0] == "Group1":
                    # faithfulness constraint
                    grp2idx = contents.index("Group2")
                    emptyidx = contents.index("")
                    grp1 = contents[1:grp2idx]
                    grp2 = contents[grp2idx + 1:emptyidx]
                    rules[colname] = {
                        ctype: f,
                        g1: list(grp1),
                        g2: list(grp2)
                    }
                elif contents[0] == "" or re.match("\d+", str(contents[0])):
                    # violations listed explicitly
                    pass
                else:
                    # markedness constraint
                    emptyidx = contents.index("")
                    grp1 = contents[:emptyidx]
                    rules[colname] = {
                        ctype: m,
                        g1: list(grp1)
                    }

            tableaux = {}

            cur_input = ""
            cur_tableau = {}
            for idx, row in df.iterrows():
                if row["input"] != "":
                    if len(cur_tableau.keys()) > 0:
                        # save previous input's tableau
                        tableaux[cur_input] = cur_tableau
                    # start a new tableau
                    cur_input = row["input"]
                    cur_tableau = {}
                cur_candidate = row["candidate"]
                cur_frequency = row["frequency"]
                if cur_frequency == "":
                    cur_frequency = 0
                else:
                    cur_frequency = float(cur_frequency)

                cur_violations = getviolations(cur_input, cur_candidate, list(df.columns[3:]), row[3:], rules)
                cur_tableau[cur_candidate] = {}
                cur_tableau[cur_candidate]["frequency"] = cur_frequency
                cur_tableau[cur_candidate]["violations"] = cur_violations

            # save the final input's tableau
            tableaux[cur_input] = cur_tableau

        self.constraints = list(df.columns[3:])

        # initialize constraint set and weights
        if len(INIT_WEIGHTS.keys()) > 0:
            # initial weights have been specified; use them
            for con in self.constraints:
                self.weights[con] = INIT_WEIGHTS[con]
                # in theory could just assign the dictionary wholesale but order could be relevant somewhere else...
        else:
            # no initial weights have been specified; start from scratch
            for con in self.constraints:
                if con.startswith("Id"):  # or a bunch of other stuff, but this is the only kind relevant to me
                    # it's a faith constraint
                    self.weights[con] = INIT_F
                else:
                    # it's a markedness constraint
                    self.weights[con] = INIT_M

        return tableaux

    # actually run the learning simulation
    def train(self):
        # put headers into history file
        headertowrite = "trial num" + "\t" + "generated" + "\t" + "heard"
        startvalstowrite = "" + "\t" + "" + "\t" + ""
        headertowrite += "".join(["\t" + c + "\tnow" for c in self.constraints]) + "\n"
        startvalstowrite += "".join(["\t\t" + str(self.weights[c]) for c in self.constraints]) + "\n"
        with io.open(self.historyfile, "w") as history:
            history.write(headertowrite)
            history.write(startvalstowrite)

            # do any necessary shuffling re a priori rankings right away
            if self.specgenbias > 0 and round(self.weights[IdBkSyl1], 10) < round(self.weights[IdBkRt] + self.specgenbias, 10):
                apriori_adjust = self.weights[IdBkRt] + self.specgenbias - self.weights[IdBkSyl1]
                self.weights[IdBkSyl1] = self.weights[IdBkRt] + self.specgenbias

                linetowrite = "" + "\t" + "a priori" + "\t" + IdBkSyl1 + ">>" + IdBkRt + "\t"
                for con in self.constraints:
                    if con != IdBkSyl1 or self.specgenbias == 0:
                        linetowrite += "\t\t"
                    else:  # specgenbias != 0
                        linetowrite += str(apriori_adjust) + "\t" + str(self.weights[IdBkSyl1])
                linetowrite += "\n"
                history.write(linetowrite)

            learningtrial = 0  # lap_count = 0
            for batchnum in range(len(LEARNING_TRIALS)):
                print("batch #", batchnum)

                # # the commented-out section below is not aligned with OTSoft in that
                # # (a) it assumes "learning trials" is the number of times through ALL of the data,
                # #   rather than the number of individual learning trials
                # # (b) it samples without replacement (ie, ensures that every data point gets seen), rather than
                # #   sampling with replacement (ie, learning proportions might be different from input file frequencies)
                # for learningtrial in range(LEARNING_TRIALS[batchnum]):
                #     lap_count += 1
                #     if learningtrial % 1000 == 0:
                #         print("trial #", learningtrial)
                #     shuffled_tableaux = random.sample(self.training_tableaux_list, len(self.training_tableaux_list))
                #     for t in shuffled_tableaux:
                #         self.learn(t, LEARNING_R_F[batchnum], LEARNING_R_M[batchnum], LEARNING_NOISE_F[batchnum], LEARNING_NOISE_M[batchnum], lap_count, history)

                # the section below replaces the one from above
                # in this version, I sample with replacement AND use the "learning trials" parameter to refer
                # to individual trials rather than number of loops through all data... so, more like OTSoft behaviour
                sampled_tableaux = random.choices(self.training_tableaux_list, k=LEARNING_TRIALS[batchnum])

                # and this is "use exact proportions"-ish
                # timestouselist = LEARNING_TRIALS[batchnum] / len(self.training_tableaux_list)
                # timesthrough = 1
                # sampled_tableaux = []
                # while timesthrough < timestouselist:
                #     sampled_tableaux += random.sample(self.training_tableaux_list, len(self.training_tableaux_list))
                #     timesthrough += 1
                # remainingsamples = LEARNING_TRIALS[batchnum] - len(sampled_tableaux)
                # sampled_tableaux += random.sample(self.training_tableaux_list, remainingsamples)

                print("number of sampled tableaux in batch", batchnum, "is", len(sampled_tableaux))
                for t in sampled_tableaux:
                    learningtrial += 1
                    if learningtrial % 5000 == 0:
                        print("trial #", learningtrial)
                    self.learn(t, LEARNING_R_F[batchnum], LEARNING_R_M[batchnum], LEARNING_NOISE_F[batchnum], LEARNING_NOISE_M[batchnum], learningtrial, history)

                print(LEARNING_TRIALS[batchnum], " trials complete")

            summarytowrite = "TOTAL\t" + "\t" + "".join(["\t\t" + str(self.weights[c]) for c in self.constraints]) + "\n"
            history.write(summarytowrite)

    # apply noise to current constraint values, for evaluation purposes
    def getevalweights(self, noise_f, noise_m):
        evalweights = {}
        for con in self.constraints:
            noise = noise_m
            if con.startswith("Id"):
                noise = noise_f
            evalweights[con] = np.random.normal(loc=self.weights[con], scale=noise)
        return evalweights

    # update constraint values given optimal candidate, intended winner, and current plasticities / biases
    def updateweights(self, tableau_df, intendedwinner, generatedoutput, cur_R_F, cur_R_M, lap_count, historystream):
        winner_df = tableau_df[tableau_df[tableau_df.columns[0]] == intendedwinner]
        optimal_df = tableau_df[tableau_df[tableau_df.columns[0]] == generatedoutput]

        adjustments = {}
        promotion_ratio = 1

        # demotion amount as usual
        # promotion amount = (# constraints demoted)/(1 + # constraints promoted)
        numpromoted = 0
        numdemoted = 0
        for c in self.constraints:
            w = winner_df[c].values[0]
            o = optimal_df[c].values[0]
            if w > 0 and o > 0:
                # cancel out violations - just look at relative difference
                overlap = min([w, o])
                w -= overlap
                o -= overlap
            if w > 0:
                numdemoted += 1  # for Magri update
                adjustments[c] = -1 * (cur_R_F if c.startswith("Id") else cur_R_M)
            elif o > 0:
                numpromoted += 1  # for Magri update
                adjustments[c] = 1 * (cur_R_F if c.startswith("Id") else cur_R_M)
        if self.magri:
            promotion_ratio = numdemoted / (1 + numpromoted)
            if numdemoted == 0:
                print(winner_df)
                print(optimal_df)

        linetowrite = str(lap_count) + "\t" + generatedoutput + "\t" + intendedwinner
        for con in self.constraints:
            if con in adjustments.keys():
                adjustment_amount = adjustments[con]
                if adjustment_amount > 0:
                    adjustment_amount *= promotion_ratio
                linetowrite += "\t" + str(adjustment_amount)
                self.weights[con] = self.weights[con] + adjustment_amount
                linetowrite += "\t" + str(self.weights[con])
            else:
                linetowrite += "\t\t"
        linetowrite += "\n"

        historystream.write(linetowrite)

        if self.specgenbias > 0 and round(self.weights[IdBkSyl1], 10) < round(self.weights[IdBkRt] + self.specgenbias, 10):
            apriori_adjust = self.weights[IdBkRt] + self.specgenbias - self.weights[IdBkSyl1]
            self.weights[IdBkSyl1] = self.weights[IdBkRt] + self.specgenbias

            linetowrite = "" + "\t" + "a priori" + "\t" + IdBkSyl1 + ">>" + IdBkRt + "\t"
            for con in self.constraints:
                if con != IdBkSyl1 or self.specgenbias == 0:
                    linetowrite += "\t\t"
                else:  # specgenbias != 0
                    linetowrite += str(apriori_adjust) + "\t" + str(self.weights[IdBkSyl1])
            linetowrite += "\n"
            historystream.write(linetowrite)

    # learn from one datum in the input file / update constraint values if an error is generated
    def learn(self, tableau_df, cur_R_F, cur_R_M, cur_noise_F, cur_noise_M, lap_count, historystream):
        # select a learning datum from distribution (which could just be all one form)
        ur = tableau_df.columns[0]
        datum = ""
        candidates = tableau_df[ur].values
        frequencies = tableau_df["frequency"].values
        frequencysum = sum(frequencies)
        frequencies = [f/frequencysum for f in frequencies]
        sample = random.uniform(0, 1)
        cumulative_freq = 0
        idx = 0
        while idx < len(frequencies) and datum == "":
            cumulative_freq += frequencies[idx]
            if sample <= cumulative_freq:
                datum = candidates[idx]
            idx += 1

        # generate the optimal candidate based on current constraint weights (ranking), with or without noise
        evalweights = self.getevalweights(cur_noise_F, cur_noise_M)
        optimal_cand = evaluate_one(tableau_df, evalweights)

        # if the optimal candidate matches the intended winner, do nothing

        # if the optimal candidate does not match the intended winner, update the weights
        if datum != optimal_cand:
            self.updateweights(tableau_df, datum, optimal_cand, cur_R_F, cur_R_M, lap_count, historystream)

    # evaluate inputs numtimes times, for each form inferring a ranking from the
    # current constraint values (along with noise)
    def testgrammar(self, numtimes):
        forms = {}  # ur --> dict of [ candidate --> frequency ]
        for t in self.tableaux_list:
            # set up UR keys to track frequency of each output
            ur = t.columns[0]
            forms[ur] = {}
            for candidate in t[ur].values:
                forms[ur][candidate] = 0
        for i in range(numtimes):
            if i % 10 == 0:
                print("test lap ", i)
            shuffled_tableaux = random.sample(self.tableaux_list, len(self.tableaux_list))

            for tableau in shuffled_tableaux:
                evalweights = self.getevalweights(LEARNING_NOISE_F[-1], LEARNING_NOISE_M[-1])
                optimalout = evaluate_one(tableau, evalweights)
                ur = tableau.columns[0]
                forms[ur][optimalout] += 1
        forms_normalized = {}
        for ur in forms.keys():
            cands, freqs = zip(*list(forms[ur].items()))
            freqsum = sum(freqs)
            freqs = [f/freqsum for f in freqs]
            forms_normalized[ur] = freqs

        results_tableaux_list =[]
        for t in self.tableaux_list:
            results_t = t.copy()
            results_t.insert(2, "outputfrequency", forms_normalized[t.columns[0]])
            results_tableaux_list.append(results_t)

        return results_tableaux_list


# end of class Learner #

# determine the violation profile for a particular candidate of a particular input
def getviolations(ur, candidate, cons, cellvalues, rules):
    violations = []
    for idx, cell in enumerate(cellvalues):
        numviolations = 0
        if re.match("\d+", str(cell)):
            # number of violations was explicitly assigned
            numviolations = int(cell)
        else:  # violation mark(s) hasn't been explicitly assigned
            constraint = cons[idx]
            if constraint in rules.keys():
                if rules[constraint][ctype] == f:  # it's a faith constraint
                    # do this twice: group1 --> group2, and then reverse
                    for direction in [0, 1]:
                        grp1 = rules[constraint][g1]
                        grp2 = rules[constraint][g2]
                        if direction == 1:
                            grp2 = rules[constraint][g1]
                            grp1 = rules[constraint][g2]
                        for in_substr in grp1:
                            numinstances = ur.count(in_substr)
                            i = -1
                            while numinstances > 0:
                                i = ur.index(in_substr, i+1)
                                cand_substr = candidate[i:i+len(in_substr)]
                                if cand_substr in grp2:
                                    numviolations += 1
                                numinstances -= 1
                else:  # it's a markedness constraint
                    numviolations = 0
                    for substring in rules[constraint][g1]:
                        numviolations += candidate.count(substring)
            else:
                # it's just empty
                numviolations = 0
        violations.append(numviolations)
    return violations


# returns a list of dataframes, where each df represents one tableau from the input file and
# has format as below in get_tableau()
# tableax = dictionary of inputstring --> { dictionary of candidate --> list of violations }
def get_tableaux(tableaux, constraints):
    list_of_dfs = []
    for ur in tableaux.keys():
        list_of_dfs.append(get_tableau(ur, tableaux[ur], constraints))
    return list_of_dfs


# returns a dataframe for a particular input (ur), where each row is a candidate, its frequency, and
# its violation profile
# tableau = dictionary of candidate --> list of violations
def get_tableau(ur, tableau, constraints):
    df_lists = []
    for cand in tableau.keys():
        df_lists.append([cand]+[tableau[cand]["frequency"]]+tableau[cand]["violations"])
    df = pd.DataFrame(df_lists, columns=[ur]+["frequency"]+constraints)
    return df


# evaluate one input's tableau given the current constraint values (any relevant noise already applied)
# evalweights is a dictionary of constraint names --> evaluation weights
def evaluate_one(tableau_df, evalweights):

    ur = tableau_df.columns[0]
    candidate_contenders = [cand for cand in tableau_df[ur].values]

    wts = list(evalweights.items())  # make it a list of key-value pairs (tuples)
    wts.sort(key=lambda x: x[1], reverse=True)
    ranking = [c for (c, w) in wts]

    winner = ""
    idx = 0
    while winner == "" and idx < len(ranking):
        c = ranking[idx]
        violns = {}
        for bla, row in tableau_df.iterrows():
            viol = row[c]
            if viol not in violns.keys():
                violns[viol] = []
            violns[viol].append(row[ur])

        violations_category = 0
        existing_violn_numbers = sorted(list(violns.keys()))
        reduced = False
        while violations_category <= max(existing_violn_numbers) and winner == "" and not reduced:
            if violations_category in existing_violn_numbers:
                successful_cands = violns[violations_category]
                reduced_contenders = [c for c in candidate_contenders if c in successful_cands]
                if len(reduced_contenders) > 0:
                    candidate_contenders = reduced_contenders
                    reduced = True
                    if len(candidate_contenders) == 1:
                        # we have a winner!
                        winner = candidate_contenders[0]
            violations_category += 1
        idx += 1
    return winner


# produce console output as well as a history file & results file,
# detailing one simulation run of one learner
def main():
    starttime = datetime.now()
    learner = Learner(FILE, magri=MAGRI, specgenbias=SPECGENBIAS)
    tableaux = learner.read_input()
    learner.set_tableaux(get_tableaux(tableaux, learner.constraints))


    print("--------------- BEGIN TRAIN ---------------------")
    learner.train()

    with io.open(learner.resultsfile, "w") as rf:

        rf.write("\n--------------- PARAMETERS ---------------------\n")
        rf.write("Magri update used: " + ("yes" if MAGRI else "no") + "\n")
        rf.write("specific > general bias: " + (str(SPECGENBIAS) if SPECGENBIAS > 0 else "no") + "\n")
        rf.write("learning trials, listed by batch: " + str(LEARNING_TRIALS) + "\n")
        rf.write("markedness plasticity, listed by batch: " + str(LEARNING_R_M) + "\n")
        rf.write("markedness noise, listed by batch: " + str(LEARNING_NOISE_M) + "\n")
        rf.write("faithfulness plasticity, listed by batch: " + str(LEARNING_R_F) + "\n")
        rf.write("faithfulness noise, listed by batch: " + str(LEARNING_NOISE_F) + "\n")
        if len(INIT_WEIGHTS.keys()) > 0:
            rf.write("initial weights: " + str(INIT_WEIGHTS) + "\n")
        else:
            rf.write("initial markedness weights = " + str(INIT_M) + "; initial faithfulness weights = " + str(INIT_F))
        rf.write("\n")

        print("\n--------------- RESULTS ---------------------\n")
        rf.write("\n--------------- RESULTS ---------------------\n\n")
        finalweights = list(learner.weights.items())
        finalweights.sort(key=lambda x: x[1], reverse=True)
        cons, weights = zip(*finalweights)
        for idx, con in enumerate(cons):
            print(con + "\t" + str(weights[idx]))
            rf.write(con + "\t" + str(weights[idx]) + "\n")

        print("\n--------------- BEGIN TEST ---------------------\n")
        rf.write("\n--------------- BEGIN TEST ---------------------\n\n")
        testresults = learner.testgrammar(100)
        for results_t in testresults:
            ordered_t = results_t.reindex([results_t.columns[0]]+list(results_t.columns[1:3])+list(cons), axis=1)
            print(ordered_t)
            rf.write(ordered_t.to_string(index=False) + "\n\n")

        endtime = datetime.now()
        print("time elapsed", endtime-starttime)
        rf.write("time elapsed: " + str(endtime-starttime))


if __name__ == "__main__":
    main()
