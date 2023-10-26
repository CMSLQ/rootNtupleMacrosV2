#!/usr/bin/env python3

import sys
import math
from tabulate import tabulate

from ROOT import TH1D, TGraphAsymmErrors, TCanvas, gROOT, TLatex, TFile
from plot_class import GetFile, GetHisto

gROOT.SetBatch()

# filePath = "/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/nanoV7/2016/analysis/eejj_20oct2020_optFinalSels/condor/analysisClass_lq_eejj___{}/output/analysisClass_lq_eejj___{}_0.root"
# filePath = "/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/nanoV7/2016/analysis/eejj_26oct2020_optFinalSels_noPdfReweight/condor/analysisClass_lq_eejj___{}/output/analysisClass_lq_eejj___{}_0.root"
# filePath = "$LQDATA/nanoV7/2017/analysis/prefire_eejj_23oct2020_optFinalSels/condor/analysisClass_lq_eejj___{}/output/analysisClass_lq_eejj___{}_0.root"
# filePath = "$LQDATA/nanoV7/2016/analysis/eejj_9nov2020_optFinalSelsOld/condor/analysisClass_lq_eejj___{}/output/analysisClass_lq_eejj___{}_0.root"
#
# filePath = "$LQDATA/nanoV7/2016/analysis/precomputePrefire_looserPSK_eejj_12apr2021_oldOptFinalSels/output_cutTable_lq_eejj/"
# filePath = "$LQDATA/nanoV7/2017/analysis/precomputePrefire_looserPSK_eejj_12apr2021_oldOptFinalSels/output_cutTable_lq_eejj/"
#filePath = "$LQDATA/nanoV7/2018/analysis/precomputePrefire_looserPSK_eejj_12apr2021_oldOptFinalSels//output_cutTable_lq_eejj/"
#filePath = "$LQDATA/nanoV7/2016/analysis/eejj_finalSelsZPL_1sep2021/output_cutTable_lq_eejj/"
#filePath = "$LQDATA/nanoV7/2016/analysis/eejj_MasymFinalSelsPunzi_31aug2021/output_cutTable_lq_eejj_MasymTest/"
#filePath = "$LQDATA/nanoV7/2016/analysis/eejj_finalSelsPunzi_2sep2021/output_cutTable_lq_eejj/"
#filePath = "$LQDATA/nanoV7/2016/analysis/eejj_MasymFinalSels_13aug2021/output_cutTable_lq_eejj_MasymTest/"
#filePath = "$LQDATA/nanoV7/2016/analysis/eejj_finalSelsPunziAddFlatMasym_3sep2021/output_cutTable_lq_eejj_MasymTest/"
#filePath += "analysisClass_lq_eejj___{}.root"
#filePath = "root://eoscms//store/group/phys_exotica/leptonsplusjets/lq/scooper/ultralegacy/analysis/2016postvfp/eejj_14jun2023_heep_bdtpunziopt/cuttable_lq_eejj_bdt/condor/"
if len(sys.argv) < 2:
    print("ERROR: Did not find path to signal root files. Usage: python plotSignalEfficiencyTimesAcceptance.py filePath")
    print("\tfilePath typically ends with 'condor', e.g., root://eoscms//store/group/phys_exotica/leptonsplusjets/lq/scooper/ultralegacy/analysis/2016postvfp/eejj_14jun2023_heep_bdtpunziopt/cuttable_lq_eejj_bdt/condor/")
    exit(-1)
filePath = sys.argv[1]
if not filePath.endswith("/"):
    filePath += "/"

# LQToDEle
#signalNameTemplate = "LQToDEle_M-{}_pair_pythia8"
signalNameTemplate = "LQToDEle_M-{}_pair_bMassZero_TuneCP2_13TeV-madgraph-pythia8"
#mass_points = [i for i in range(300, 3100, 100)]  # go from 300-3000 in 100 GeV steps
#mass_points.extend([3500, 4000])
mass_points = [i for i in range(1000, 2100, 100)]
if "2016preVFP" in filePath:
    signalNameTemplate = "LQToDEle_M-{}_pair_bMassZero_TuneCP2_13TeV-madgraph-pythia8_APV"
filePath += signalNameTemplate + "/" + signalNameTemplate + "_0.root"

# LQToUE
# signalNameTemplate = "LQToUE_M-{}_BetaOne_pythia8"
# mass_points = [i for i in range(300, 2100, 100)]  # go from 300-2000 in 100 GeV steps
#
doPDFReweight = False

if "2016preVFP" in filePath:
    year = "2016preVFP"
elif "2016postVFP" in filePath:
    year = "2016postVFP"
elif "2017" in filePath:
    year = "2017"
elif "2018" in filePath:
    year = "2018"
if "UE" in signalNameTemplate:
    signalNameShort = "eeuu"
elif "DEle" in signalNameTemplate:
    signalNameShort = "eedd"

# histNameBase = "histo1D__{}__EventsPassingCuts"
histName = "EventsPassingCuts"
totalEventsByMass = []
eventsAtFinalSelByMass = []
firstFinalSelBin = 39

histTotal = TH1D("total", "total", 38, 250, 4050)
histTotal.Sumw2()
histPass = TH1D("pass", "pass", 38, 250, 4050)
histPass.Sumw2()

for i, mass in enumerate(mass_points):
    # sampleName = signalNameTemplate.format(mass)
    # filename = filePath.format(sampleName, sampleName)
    filename = filePath.format(mass, mass)
    tfile = GetFile(filename)
    # histName = histNameBase.format(sampleName)
    eventsPassingHist = GetHisto(histName, tfile)
    # noCutEntries = hist.GetBinContent(1)
    sumOfWeightsHist = GetHisto("SumOfWeights", tfile)
    sumWeights = sumOfWeightsHist.GetBinContent(1)
    sumWeightsErr = sumOfWeightsHist.GetBinError(1)
    # print("For file={}, hist={}, got sumweights = {}".format(filename, sumOfWeightsHist.GetName(), sumWeights), flush=True)
    lhePdfWeightsHist = tfile.Get("LHEPdfSumw")
    lhePdfWeightSumw = lhePdfWeightsHist.GetBinContent(1)  # sum[genWeight*pdfWeight_0]
    # print hist.GetXaxis().GetBinLabel(2)
    # finalSelName = "min_M_ej_LQ{}".format(mass)
    finalSelName = "BDTOutput_LQ{}".format(mass)
    finalSelBin = eventsPassingHist.GetXaxis().FindBin(finalSelName)
    # print("For file={}, hist={}, found bin for {} = {}".format(filename, eventsPassingHist.GetName(), finalSelName, finalSelBin), flush=True)
    # merged hists have no bin labels; have to use super ugly hack
    # finalSelBin = firstFinalSelBin+(i*3)
    # print "mass={}, finalSelBin ={}".format(mass, finalSelBin)
    # finalSelEntries = eventsPassingHist.GetBinContent(finalSelBin)
    if eventsPassingHist.ClassName() == "TProfile":
        binContent = eventsPassingHist.GetBinContent(finalSelBin)*eventsPassingHist.GetBinEntries(finalSelBin)
        binError = math.sqrt(eventsPassingHist.GetSumw2().At(finalSelBin))
        firstBinContent = eventsPassingHist.GetBinContent(1)*eventsPassingHist.GetBinEntries(1)
        firstBinError = math.sqrt(eventsPassingHist.GetSumw2().At(1))
    else:
        binContent = eventsPassingHist.GetBinContent(finalSelBin)
        binError = eventsPassingHist.GetBinError(finalSelBin)
        firstBinContent = eventsPassingHist.GetBinContent(1)
        firstBinError = eventsPassingHist.GetBinError(1)
    # totalEventsByMass.append(round(noCutEntries, 3))
    totalEventsByMass.append(round(sumWeights, 3))
    # print("LQ {}: final sel bin {}, binContent={}, binEntries={}, passing={} +/- {}, total = {} +/- {}, firstBinContent = {}, sumWeightsErr = {}".format(
    #         mass, finalSelBin, eventsPassingHist.GetBinContent(finalSelBin), eventsPassingHist.GetBinEntries(finalSelBin), binContent, binError, round(sumWeights, 3), firstBinError, firstBinContent, sumWeightsErr))
    if "2016" in filename and doPDFReweight:
        if "LQToBEle" in filename or "LQToDEle" in filename:
            totalEventsByMass.pop()
            totalEventsByMass.append(round(lhePdfWeightSumw, 3))
            print("\tapplying LHEPdfWeight={} to dataset={}".format(
                    lhePdfWeightSumw, filename)+"[instead of original sumWeights={}]".format(sumWeights))

    eventsAtFinalSelByMass.append(round(binContent, 4))
    histTotal.SetBinContent(histTotal.FindBin(mass), sumWeights)
    # histTotal.SetBinError(histTotal.FindBin(mass), firstBinError)
    # approximation: what we do here is take the relative stat error from the "raw" number of MC events and apply it to the sumWeights
    # FIXME: update this to use the bin error; a new round of skims should have the correct error as sqrt(sumgenw2)
    histTotal.SetBinError(histTotal.FindBin(mass), sumWeights*firstBinError/firstBinContent)
    histPass.SetBinContent(histPass.FindBin(mass), binContent)
    histPass.SetBinError(histTotal.FindBin(mass), binError)
    tfile.Close()


# print "masses:", mass_points
# print "total:", totalEventsByMass
# print "pass:", eventsAtFinalSelByMass
# print "effAcc:", [n/d for n, d in zip(eventsAtFinalSelByMass, totalEventsByMass)]
table = []
for idx, mass in enumerate(mass_points):
    eventsAtFinalSel = eventsAtFinalSelByMass[idx]
    total = totalEventsByMass[idx]
    try:
        effAcc = eventsAtFinalSel/total
    except ZeroDivisionError as e:
        raise RuntimeError("Got ZeroDivisionError for mass={}; eventsAtFinalSel={} / total={}".format(mass, eventsAtFinalSel, total))
    row = [mass, eventsAtFinalSel, total, effAcc*100.0]
    table.append(row)

print()
print(tabulate(table, headers=["Mass", "Passing", "Total", "eff*acc [%]"], tablefmt="github", floatfmt=".2f"))
print()

# tcan2 = TCanvas()
# tcan2.cd()
# histTotal.Draw()

# tcan3 = TCanvas()
# tcan3.cd()
# histPass.Draw()

tcan = TCanvas()
tcan.cd()
graph = TGraphAsymmErrors()
graph.Divide(histPass, histTotal, "cl=0.683 b(1,1) modev")
# graph.Divide(histPass, histTotal, "cp")
graph.Draw("ap")
graph.GetYaxis().SetRangeUser(0, 0.7)
graph.GetYaxis().SetTitle("Acceptance #times efficiency")
graph.GetYaxis().SetTitleFont(42)
graph.GetYaxis().SetLabelFont(42)
# graph.GetYaxis().SetLabelOffset(0.007)
# graph.GetYaxis().SetLabelSize(0.06)
# graph.GetYaxis().SetTitleOffset(1.)
# graph.GetYaxis().SetTitleSize(0.07)
graph.GetYaxis().CenterTitle(1)
#
graph.GetXaxis().SetTitle('#it{m}_{LQ} [GeV]')
graph.GetXaxis().SetTitleFont(42)
graph.GetXaxis().SetLabelFont(42)
# graph.GetXaxis().SetLabelOffset(0.01)
graph.GetXaxis().SetTitleOffset(1.)
# graph.GetXaxis().SetLabelSize(0.06)
# graph.GetXaxis().SetTitleSize(0.07)
graph.GetXaxis().SetNdivisions(505)
graph.SetName("accTimesEffGraph")
tcan.Update()

# -- draw label
labelOffset = 0.03
ystart = 0.64
xstart = 0.35
hsize = 0.21
vsize = 0.25
l = TLatex()
l.SetNDC()
l.SetTextAlign(12)
l.SetTextFont(132)
# l.SetTextSize(0.065)
l.SetTextSize(0.045)
l.DrawLatex(
    xstart - hsize + 0, ystart + vsize - 0.05, "CMS Preliminary {}".format(year)
)
# l.DrawLatex(
#     xstart - hsize + 0, ystart + vsize - 0.10, "#it{Simulation}"
# )
l.DrawLatex(
    xstart - hsize + 0, ystart + vsize - 0.10, "Scalar LQ #bar{LQ} #rightarrow "+signalNameShort
)
tcan.Update()
tcan.Print("accTimesEff.pdf")

outFile = TFile("accTimesEff.root", "recreate")
tcan.Write()
graph.Write()
outFile.Close()
