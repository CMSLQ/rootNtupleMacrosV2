#!/usr/bin/env python3

import sys
import math
import copy
import numpy as np
import ctypes
from optparse import OptionParser
from plot_class import GetFile, Plot, Plot2D, generateHistoList, generateHisto, generateHistoBlank, generateHistoListFakeSystsFromModel, makeTOC, QuasiRebinHisto, RenormalizeQCDHistoNormsAndUncs
from ROOT import gROOT, kCyan, kRed, TCanvas, TGraphAsymmErrors, TH1D, TFile
import cmsstyle as CMS


def SeparateArgs(arg):
    split = arg.split(",")
    return split


gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")
# gErrorIgnoreLevel = kWarning  # doesn't work

usage = "usage: %prog [options] \nExample: \npython makeStackHistoTemplateV2_eejj.py -q qcdFilePath -m mcDataFilePath -y years [-s signalName] [--fitDiagFilepath path] [--postFitJSON json] [--preFit] [--postFit] [--fitType postFitType]"
parser = OptionParser(usage=usage)

parser.add_option(
    "-q",
    "--qcdFilePath",
    dest="qcdFilePath",
    help="full path to QCD plot files",
    metavar="QCDFILEPATH",
)

parser.add_option(
    "-m",
    "--mcDataFilePath",
    dest="mcDataFilePath",
    help="full path to MC/data plot files",
    metavar="MCDATAFILEPATH",
)

parser.add_option(
    "-y",
    "--years",
    dest="years",
    help="analysis years",
    metavar="YEARS",
)
    
parser.add_option(
    "-s",
    "--signalName",
    dest="signalName",
    default="LQToDEle",
    help="signal name (defaults to LQToDEle)",
    metavar="SIGNALNAME",
)

parser.add_option(
    "--fitDiagFilepath",
    dest="fitDiagFilepath",
    default=None,
    help="FitDiagnostics root files path (from combine output)",
    metavar="FITDIAGFILEPATH",
)

parser.add_option(
    "--postFitJSON",
    dest="postFitJSON",
    default=None,
    help="postfit JSON file (from fitdiagnostics output)",
    metavar="POSTFITJSON",
)

parser.add_option(
    "--postFit",
    dest="postFit",
    default=False,
    action="store_true",
    help="do postfit final selection plots",
    metavar="POSTFIT",
)

parser.add_option(
    "--preFit",
    dest="preFit",
    default=False,
    action="store_true",
    help="do prefit final selection plots",
    metavar="PREFIT",
)

parser.add_option(
    "--fitType",
    dest="fitType",
    default=None,
    help="fit type for postfit final selection plots",
    metavar="FITTYPE",
)

parser.add_option(
    "--reducedSet",
    dest="reducedSet",
    default=False,
    action="store_true",
    help="produce reduced set of plots",
    metavar="REDUCEDSET",
)
    
parser.add_option(
    "--preselection",
    dest="preselection",
    default=False,
    action="store_true",
    help="produce preselection plots",
    metavar="PRESELECTION",
)

# parser.add_option(
#     "--preliminary",
#     dest="preliminary",
#     default=False,
#     action="store_true",
#     help="produce plots with preliminary label",
#     metavar="preliminary",
# )

(options, args) = parser.parse_args()

requiredOpts = [options.qcdFilePath, options.mcDataFilePath, options.years]
if any(opt is None for opt in requiredOpts):
    print("ERROR: one or more required options not given.")
    raise RuntimeError(usage)

dataMCFilePath = options.mcDataFilePath.rstrip("/") + "/"
signalName = options.signalName
if "LQToDEle" in signalName:
    signalSampleName = "LQToDEle_M-{}_pair"
elif "LQToBEle" in signalName:
    signalSampleName = "LQToBEle_M-{}_pair"
else:
    raise RuntimeError("Couldn't understand provided signal name '{}'. Must contain LQToDEle or LQToBEle.".format(signalName))
print("INFO: using {} as signal name".format(signalName))
 
inputFileQCDPath = options.qcdFilePath


doSystematics = True
doQCD = True
quasiRebinQCD = False  # TODO: not sure the current algorithm works correctly or is optimal
dyjNormSyst = 0.2
ttbarNormSyst = 0.1
qcdNormSyst = 0.4
doPreselPlots = options.preselection
doBTagPlots = False
doFinalSelectionPlots = not doPreselPlots
doFullSet = not options.reducedSet
blindFinalSelectionData = False
doPrefit = options.preFit
doPostfit = options.postFit
if doPrefit or doPostfit:
    doPreselPlots = False
    print("INFO: Not doing preselection plots as prefit or postfit option is enabled.")
if doPrefit:
    options.fitType = "prefit"
if doPostfit:
    if options.fitType is None:
        print(usage)
        raise RuntimeError("Must specity fitType when requesting postfit scaling.")
do2016 = False
do2016pre = False
do2016post = False
do2017 = False
do2018 = False
doRunII = False
for year in SeparateArgs(options.years):
    if '2016preVFP' in year:
        do2016 = True
        do2016pre = True
        qcdYearsToUse = ['2016preVFP']
        lumiNormSyst = 0.012
    elif '2016postVFP' in year:
        do2016 = True
        do2016post = True
        qcdYearsToUse = ['2016postVFP']
        lumiNormSyst = 0.012
    elif '2016' in year:
        do2016 = True
        qcdYearsToUse = ['2016preVFP', '2016postVFP']
        lumiNormSyst = 0.012
    elif '2017' in year:
        do2017 = True
        qcdYearsToUse = ['2017']
        lumiNormSyst = 0.0082
    elif '2018' in year:
        do2018 = True
        qcdYearsToUse = ['2018']
        lumiNormSyst = 0.0084
    elif 'RunII' == year or 'all' == year.lower():
        doRunII = True
        qcdYearsToUse = ['2016preVFP', '2016postVFP', '2017', '2018']
        lumiNormSyst = 0.0073
    else:
        print("ERROR: could not find one of 2017/2017/2018/RunII/all in given year. cannot do year-specific customizations. quitting.")
        exit(-1)

# LQmasses = [700, 1500]  # new samples don't have 650 GeV point
#LQmasses = [1500, 2000]  # new samples don't have 650 GeV point
# LQmasses = [1000, 1200, 1500]
LQmasses = [1000, 1500]
# LQmassesFinalSelection = [1200, 1500]
# LQmassesFinalSelection = list(range(300, 3100, 100))
LQmassesFinalSelection = [800]
# LQmassesFinalSelection = [1200]
if not doFinalSelectionPlots:
    LQmassesFinalSelection = []

zUncBand = "no"
makeRatio = 1
makeNSigma = 1

pt_rebin = 2

if doSystematics:
    systNames = ["Prefire", "EES", "EER", "JES", "JER", "EleRecoSF", "EleIDSF", "EleTrigSF", "Pileup", "UnclusteredEne", "LHEPdf", "LHEScale"]
    uncorrelatedSysts = ["Prefire", "EES", "EER", "JES", "JER", "EleRecoSF", "EleIDSF", "EleTrigSF", "Pileup", "UnclusteredEne"]
    if doPrefit or doPostfit:
        systNames = ["TotalSystematic"]
    print("INFO: using systNames={}".format(systNames))
else:
    systNames = []
    print("INFO: not including systematics on plots")
    #FIXME: need to also handle case when we have the 2-D syst hists but don't want to plot the systs for some reason

histoBaseName = "histo1D__SAMPLE__cutHisto_allOtherCuts___________VARIABLE"
histoBaseName_userDef = "histo1D__SAMPLE__VARIABLE"

histoBaseName2D = "histo2D__SAMPLE__cutHisto_allOtherCuts___________VARIABLE"
histoBaseName2D_userDef = "histo2D__SAMPLE__VARIABLE"

samplesForStackHistos_QCD = []
if doQCD:
    # samplesForStackHistos_QCD = ["QCD_EMEnriched"]
    samplesForStackHistos_QCD = ["QCDFakes_DATA"]

# print "Using QCD plot file: {}".format(inputFileQCD)
# samplesForStackHistos_ZJets  = [ "TTbar_Madgraph", "ZJet_amcAtNLO_Inc" ]
# samplesForStackHistos_other = [ "WJet_amcAtNLO_Inc" , "SingleTop", "QCD_EMEnriched", "DIBOSON"]
# samplesForStackHistos_other = [ "WJet_amcAtNLO_Inc" , "SingleTop", "PhotonJets_Madgraph", "QCD_EMEnriched", "DIBOSON"]

# amc@NLO
# samplesForStackHistos_ZJets  = [ "TTbar_amcatnlo_Inc", "ZJet_amcatnlo_Inc" ]
# samplesForStackHistos_other = [ "OTHERBKG_amcAtNLOInc" ]
# keysStack             = [ "Other backgrounds", "t#bar{t} (MG5_aMC)"  ,  "Z/#gamma* + jets (MG5_aMC)"  ]

# MG Inc
# samplesForStackHistos_other = [ "OTHERBKG_MGInc" ]
# samplesForStackHistos_ZJets  = [ "TTbar_Madgraph", "ZJet_Madgraph_Inc" ]
# keysStack             = [ "Other backgrounds", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (MG Inc)"  ]

#useMGHT = False
#if useMGHT:
#    # MG HT
#    samplesForStackHistos_other = ["OTHERBKG_MG_HT"]
#    samplesForStackHistos_ZJets = ["TTbar_Madgraph", "ZJet_Madgraph_HT"]
#    # keysStack             = [ "Other backgrounds", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (MG HT)"  ]
#    keysStack = [
#        "Other backgrounds",
#        "QCD multijet",
#        "t#bar{t} (Madgraph)",
#        "Z/#gamma* + jets (MG HT)",
#    ]
#
## amc@NLO Pt Z, TTBar MG
# samplesForStackHistos_ZJets  = [ "TTbar_Madgraph", "ZJet_amcatnlo_ptBinned" ]
# samplesForStackHistos_other = [ "OTHERBKG_MG_ZJetPt" ]
##keysStack             = [ "Other backgrounds", "t#bar{t} (MG)"  ,  "Z/#gamma* + jets (MG5_aMC Pt)"  ]
# keysStack             = [ "Other backgrounds", "QCD multijet", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (MG5_aMC Pt)"  ]

if do2016:
    ilumi = "36.3"
    if do2016pre:
        ilumi = "19.5"  # preVFP
    elif do2016post:
        ilumi = "16.8"  # postVFP
    samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]  # nominal
    # samplesForStackHistos_ZJets = ["ZJet_powhegminnlo"]
    # samplesForStackHistos_ZJets = ["ZJet_madgraphLO_HT"]
    #samplesForStackHistos_other = ["OTHERBKG_WJetAMCJetBinned_dibosonNLO_tribosonGJetsTTX"]  # old UL with single QCD fakes from MC
    # samplesForStackHistos_other = ["SingleTop", "DIBOSON_nlo"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_other = ["OTHERBKG_dibosonNLO_singleTop"]  # UL with single/double QCD fakes from data
    #samplesForStackHistos_ttbar = ["TTbar_powheg_all"]  # old UL with single QCD fakes from MC
    samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    #
    #samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned"]
    # samplesForStackHistos_ZJets  = [ "ZJet_amcatnlo_Inc" ]
    # MC ttbar
    # samplesForStackHistos_ttbar = [ "TTbar_amcatnlo_Inc" ]
    # keysStack             = ["QCD multijet (data)", "Other backgrounds", "t#bar{t} (MG5_aMC)"  ,  "Z/#gamma* + jets (MG5_aMC Pt)"  ]
    # data-driven ttbar
    #samplesForStackHistos_ttbar = ["TTBarFromDATA"]
    keysStack = ["QCD multijet (data)"] if doQCD else []
    keysStack.extend([
        # "QCD multijet (MC)",
        # "Other backgrounds",
        "Diboson + ST #rightarrow tW",
        # "t#bar{t} #rightarrow 2L2#nu (powheg)",
        "t#bar{t} (powheg)",
        "Z/#gamma* + jets (powheg MiNNLO)" if "ZJet_powhegminnlo" in samplesForStackHistos_ZJets[0] else "Z/#gamma* + jets (MG5_aMC Pt)",
        # "Z/#gamma* + jets (MG HT)",
    ])
elif do2017:
    ilumi = "41.5"
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_Inc"]
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # samplesForStackHistos_ZJets = ["ZToEE"]
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_jetBinned"]
    # samplesForStackHistos_ZJets = ["ZJet_jetAndPtBinned"]
    samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # MC ttbar
    # samplesForStackHistos_ttbar = ["TTbar_powheg"]
    # samplesForStackHistos_ttbar = ["TTbar_powheg_all"]  # old UL with single QCD fakes from MC
    # samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    # samplesForStackHistos_ZJets = ["ZJet_powhegminnlo"]
    # samplesForStackHistos_other = ["SingleTop", "DIBOSON_nlo"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_other = ["OTHERBKG_dibosonNLO_singleTop"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    keysStack = ["QCD multijet (data)"] if doQCD else []
    keysStack.extend([
        # "QCD multijet (MC)",
        "Other backgrounds",
        "t#bar{t} (powheg)",
        # "Z/#gamma* + jets (MG5_aMC Inc.)",
        "Z/#gamma* + jets (MG5_aMC Pt+IncStitch)",
        # "Z/#gamma* + jets (ZToEE)",
        # "Z/#gamma* + jets (MG5_aMC jet-binned)",
        # "Z/#gamma* + jets (MG5_aMC jet/pt-binned)",
    ])
elif do2018:
    ilumi = "59.8"
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_Inc"]
    samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_jetBinned"]
    # samplesForStackHistos_ZJets = ["ZJet_jetAndPtBinned"]
    # samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # MC ttbar
    # samplesForStackHistos_ttbar = ["TTbar_powheg"]
    # samplesForStackHistos_ttbar = ["TTbar_powheg_all"]  # old UL with single QCD fakes from MC
    # samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    # samplesForStackHistos_ZJets = ["ZJet_powhegminnlo"]
    # samplesForStackHistos_other = ["SingleTop", "DIBOSON_nlo"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_other = ["OTHERBKG_dibosonNLO_singleTop"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    keysStack = ["QCD multijet (data)"] if doQCD else []
    keysStack.extend([
        # "QCD multijet (MC)",
        "Other backgrounds",
        "t#bar{t} (powheg)",
        # "Z/#gamma* + jets (MG5_aMC Inc.)",
        "Z/#gamma* + jets (MG5_aMC Pt+IncStitch)",
        # "Z/#gamma* + jets (MG5_aMC jet-binned)",
        # "Z/#gamma* + jets (MG5_aMC jet/pt-binned)",
    ])
elif doRunII:
    ilumi = "138"
    samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # samplesForStackHistos_other = ["SingleTop", "DIBOSON_nlo"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_other = ["OTHERBKG_dibosonNLO_singleTop"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    keysStack = ["QCD multijet (data)"] if doQCD else []
    keysStack.extend([
        "Other backgrounds",
        # "t#bar{t} (powheg)",
        # "Z/#gamma* + jets (MG5_aMC Pt+IncStitch)",
        "t#bar{t} + jets",
        "Z/#gamma* + jets",
    ])

#samplesForStackHistos_ZJets  = [ "TTbar_FromData", "ZJet_Madgraph" ]
# older
# samplesForStackHistos = samplesForStackHistos_other + samplesForStackHistos_ZJets
#samplesForStackHistos = samplesForHistos_QCD if doQCD else []
samplesForStackHistosMC = (
    samplesForStackHistos_other
    + samplesForStackHistos_ttbar
    + samplesForStackHistos_ZJets
)
samplesForStackHistos = [samplesForStackHistos_QCD] + samplesForStackHistosMC

dataMCFilesDict = {sample: GetFile(dataMCFilePath + "analysisClass_lq_eejj_{}_plots.root".format(sample)) for sample in samplesForStackHistosMC}
#samplesForStackHistos += samplesForStackHistos_other + samplesForStackHistos_ttbar + samplesForStackHistos_ZJets
# print 'samplesForStackHistos',samplesForStackHistos
# keysStack             = [ "Other backgrounds", "QCD multijet", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (MG HT)"  ]
# keysStack             = [ "Other backgrounds", "QCD multijet", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (amc@NLO Pt)"  ]
if doQCD:
    # stackColorIndexes = [kCyan, 9, 600, kRed]
    stackColorIndexes = [CMS.p8.kCyan, CMS.p8.kAzure, CMS.p8.kBlue, CMS.p8.kRed]
else:
    stackColorIndexes = [CMS.p8.kAzure, CMS.p8.kBlue, CMS.p8.kRed]
stackFillStyleIds = [1001, 1001, 1001, 1001]

##keysStack             = [ "Other backgrounds", "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets (MG Inc)"  ]
##stackColorIndexes     = [ 9                  , 600         ,  kRed           ]
###stackFillStyleIds     = [ 3008               , 3004        ,  3345           ]
##stackFillStyleIds     = [ 1001               , 1001        ,  1001           ]
#
##keysStack             = ["t#bar{t} (Madgraph)"     , "SingleTop"  ,"DIBOSON"   ,  "WJet(amc@NLO)",   "PhotonJets_Madgraph",  "QCD_EMEnriched",     "Z/#gamma* + jets"  ]
##stackColorIndexes     = [   600                    , kMagenta     , kGreen+3   ,  kCyan-3        ,           7            ,    kMagenta+3    ,     kRed+1           ]
##stackFillStyleIds     = [   1001                   , 1001         ,    1001    ,  1001           ,           1001         ,        1001      ,     3345           ]
##keysStack             = [ "WJet(amc@NLO)"    , "SingleTop"  ,  "QCD_EMEnriched",  "DIBOSON"   , "t#bar{t} (Madgraph)"  ,  "Z/#gamma* + jets"  ]
##stackColorIndexes     = [ 9                  , kGreen+3     ,        6         ,        3     ,    600                 ,  kRed           ]
##stackFillStyleIds     = [ 1001               , 1001         ,        1001      ,      1001    ,    3004                ,  3345           ]
#
##stackColorIndexes.reverse()
##stackFillStyleIds.reverse()

# signalSampleLabel = "LQToDEle, M={} GeV"
signalSampleLabel = "m_{{LQ}} = {} GeV, #beta = 1"
samplesForHistos = [signalSampleName.format(lqmass) for lqmass in LQmasses] + [signalSampleName.format(lqmass) for lqmass in LQmassesFinalSelection]
samplesForHistos = sorted(list(set(samplesForHistos)))
# samplesForHistos = list(samplesForHistos)
extraLQMasses = [500]
extraSamplesForHistos = [signalSampleName.format(lqmass) for lqmass in extraLQMasses]
for sample in samplesForHistos + extraSamplesForHistos:
    dataMCFilesDict[sample] = GetFile(dataMCFilePath + "analysisClass_lq_eejj_{}_plots.root".format(sample))
keys = [signalSampleLabel.format(lqmass) for lqmass in LQmasses] + [signalSampleLabel.format(lqmass) for lqmass in extraLQMasses]
# keys = list(set([signalSampleLabel.format(lqmass) for lqmass in LQmasses+LQmassesFinalSelection+extraLQMasses]))
# samplesForHistos.extend([signalSampleName.format(lqmass) for lqmass in extraLQMasses])
# no signal
# samplesForHistos = []
# keys             = []


samplesForHistos_blank = []
keys_blank = []

sampleForDataHisto = "DATA"
dataMCFilesDict[sampleForDataHisto] = GetFile(dataMCFilePath + "analysisClass_lq_eejj_DATA_plots.root")
#sampleForDataHisto = "SingleElectron_2016_HIPM"
#sampleForDataHisto = "SingleElectron_2016"
# dataBlindAbovePt1 = 800 # GeV; used for ele Pt1, Mee, Mej
# dataBlindAbovePt2 = 400 # for ele pt2
# dataBlindAboveSt = 1500 # for St plots
dataBlindAboveSt = -1  # for St plots
dataBlindAboveAll = -1
dataBlindAbovePt1 = -1  # GeV; used for ele Pt1, Mee, Mej
dataBlindAbovePt2 = -1  # for ele pt2

print("INFO: Using DYJets sample:", samplesForStackHistos_ZJets, " ("+keysStack[3] if doQCD else keysStack[2]+")")


def makeDefaultPlot(
    variableName,
    histoBaseName=histoBaseName_userDef,
    samplesForHistos=samplesForHistos,
    keys=keys,
    samplesForStackHistos=samplesForStackHistos,
    keysStack=keysStack,
    sampleForDataHisto=sampleForDataHisto,
    zUncBand=zUncBand,
    makeRatio=makeRatio,
    dataBlindAbove=-1,
    systs=False,
    systList=systNames,
    rescaleDYJTTBarSystsAtPreselection=False,
    fitDiagFilePath=None,
    postFitJSON=None,
    doPrefit=False,
    fitType=None,
    massPoint=None,
):
    plot = Plot()
    plot.postFitJSON = postFitJSON
    plot.massPoint = massPoint
    plot.fitType = fitType
    systList = []
    if systs:
        histoBaseName = histoBaseName.replace("histo1D", "histo2D")
        histoBaseName+="WithSystematics"
        systList = systNames
    plot.systNames = systList
    plot.histosStack = []
    plot.histosStack.extend(generateHistoList(
            histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other],
            1.0, True, histoBaseName.replace("SAMPLE", "OTHERBKG").replace("VARIABLE", variableName), 
            lumiSyst=lumiNormSyst,
            years=qcdYearsToUse,
            masses=LQmassesFinalSelection,
            fitDiagFilePath=fitDiagFilePath,
            postFitJSON=postFitJSON,
            doPrefit=doPrefit,
            fitType=fitType
        ))
    plot.histosStack.extend(generateHistoList(
            histoBaseName, samplesForStackHistos_ttbar, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ttbar],
            rescaleSystsAtPreselection=rescaleDYJTTBarSystsAtPreselection,
            normSyst=ttbarNormSyst,
            years=qcdYearsToUse,
            masses=LQmassesFinalSelection,
            fitDiagFilePath=fitDiagFilePath,
            postFitJSON=postFitJSON,
            doPrefit=doPrefit,
            fitType=fitType
            
        ))
    plot.histosStack.extend(generateHistoList(
            histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets],
            rescaleSystsAtPreselection=rescaleDYJTTBarSystsAtPreselection,
            normSyst=dyjNormSyst,
            years=qcdYearsToUse,
                masses=LQmassesFinalSelection,
            fitDiagFilePath=fitDiagFilePath,
                postFitJSON=postFitJSON,
            doPrefit=doPrefit,
            fitType=fitType
        ))
    qcdHists = []
    for year in qcdYearsToUse:
        qcdFile = TFile.Open(inputFileQCDPath.format(year))
        if systs:
            qcdHistsThisYear = generateHistoListFakeSystsFromModel(
                histoBaseName.replace("histo2D", "histo1D").replace("WithSystematics", ""),  # never have systs for data-driven QCD
                samplesForStackHistos_QCD,
                variableName,
                qcdFile,
                year,
                LQmassesFinalSelection,
                qcdNormSyst,
                plot.histosStack[-1],
                scale=1,
                fitDiagFilePath=fitDiagFilePath,
                postFitJSON=None,
                doPrefit=options.preFit,
                fitType=options.fitType
            ) if doQCD else []
        else:
            qcdHistsThisYear = generateHistoList(
                histoBaseName,
                samplesForStackHistos_QCD,
                variableName,
                qcdFile 
            ) if doQCD else []
        if len(qcdHists) < 1:
            qcdHists = qcdHistsThisYear
        else:
            for idx, hist in enumerate(qcdHists):
                hist.Add(qcdHistsThisYear[idx])
        qcdFile.Close()
    if postFitJSON is not None:
        if len(samplesForStackHistos_QCD) > 1:
            raise RuntimeError("Don't currently know how to handle this case where we have more than 1 QCD sample, and we have {}.".format(len(samplesForStackHistos_QCD)))
        qcdHistsNew = []
        # in this case, we simply add up the QCD hists for each year, and then rescale once
        for hist in qcdHists:
            hist = RenormalizeQCDHistoNormsAndUncs(samplesForStackHistos_QCD[0], "noyear", hist, LQmassesFinalSelection, None, None, postFitJSON, doPrefit, fitType)
            qcdHistsNew.append(hist)
        qcdHists = qcdHistsNew
    if quasiRebinQCD:
        qcdHistsNew = []
        for hist in qcdHists:
            histNew = QuasiRebinHisto(hist)
            qcdHistsNew.append(histNew)
        qcdHists = qcdHistsNew
    plot.histosStack[0:0] = qcdHists
    #plot.histosStack = generateHistoList(histoBaseName, samplesForStackHistos_ZJets, variableName, File_preselection)
    #print("2) histosStack=", plot.histosStack)
    plot.histos = generateHistoList(
        histoBaseName, samplesForHistos, variableName, [dataMCFilesDict[sample] for sample in samplesForHistos], lumiSyst=lumiNormSyst
    )
    if dataBlindAbove != 0 and sampleForDataHisto != "":
        scale = 1.0
        plot.histodata = generateHisto(
            histoBaseName,
            sampleForDataHisto,
            variableName,
            dataMCFilesDict[sampleForDataHisto],
            scale,
            dataBlindAbove,
        )
        plot.histodataBlindAbove = dataBlindAbove
        if systs:
            plot.histodata = copy.deepcopy(plot.histodata.ProjectionX(plot.histodata.GetName()+"projx", 1, 1))
    #bkgTotalHist = TH1D()
    #for index, sampleHisto in enumerate(plot.histosStack):
    #    histo = copy.deepcopy(sampleHisto)
    #    if index == 0:
    #        bkgTotalHist = histo.Clone()
    #    else:
    #        bkgTotalHist.Add(histo)
    #if bkgTotalHist.InheritsFrom("TH2"):
    newHistosStack = []
    plot.histos2DStack = []
    if plot.histosStack[-1].InheritsFrom("TH2"):
        #print()
        # print("INFO: redoing stack plots for histo={}".format(plot.histosStack[0].GetName().split("__")[-1]))
        for index, sampleHisto in enumerate(plot.histosStack):
            histo = copy.deepcopy(sampleHisto)
            if index == 0:
                plot.bkgTotalHist = histo.Clone()
            else:
                #print("INFO: attempt to merge plot {} to bkgTotalHist={}".format(histo.GetName(), plot.bkgTotalHist.GetName()))
                plot.bkgTotalHist.Add(histo)
            totSystBin = histo.GetYaxis().FindFixBin("TotalSystematic")
            if totSystBin >= 0:
                integralErr = ctypes.c_double()
                integral = histo.IntegralAndError(0, histo.GetNbinsX()+2, 1, 1, integralErr)
                systIntegralErr = ctypes.c_double()
                systIntegral = histo.IntegralAndError(0, histo.GetNbinsX()+2, totSystBin, totSystBin, systIntegralErr)
                print("INFO: For plot={}, Add histo name={} with entries={}, integral = {} +/- {}; TotalSystematic integral = {} +/- {}; total uncertainty = {}".format(
                    variableName, histo.GetName(), histo.GetEntries(), integral, integralErr, systIntegral, systIntegralErr.value, math.sqrt(integralErr.value**2 + systIntegralErr.value**2)), flush=True)
            newHistosStack.append(copy.deepcopy(histo.ProjectionX(histo.GetName()+"projx", 1, 1)))  # convert to 1-D nominal hist
            plot.histos2DStack.append(copy.deepcopy(histo))
        totSystBin = plot.bkgTotalHist.GetYaxis().FindFixBin("TotalSystematic")
        if totSystBin >= 0:
            integralErr = ctypes.c_double()
            integral = plot.bkgTotalHist.IntegralAndError(0, plot.bkgTotalHist.GetNbinsX()+2, 1, 1, integralErr)
            systIntegralErr = ctypes.c_double()
            systIntegral = plot.bkgTotalHist.IntegralAndError(0, plot.bkgTotalHist.GetNbinsX()+2, totSystBin, totSystBin, systIntegralErr)
            print("INFO: For plot={}, bkgTotalHist integral = {} +/- {}; TotalSystematic integral = {} +/- {}; total uncertainty = {}".format(
                variableName, integral, integralErr, systIntegral, systIntegralErr.value, math.sqrt(integralErr.value**2 + systIntegralErr.value**2)), flush=True)
        plot.histosStack = newHistosStack
        plot.addBkgUncBand = True
    newHistos = []
    # plot.histos2D = []
    if plot.histos[-1].InheritsFrom("TH2"):
        #print()
        #print("INFO: redoing stack plots for histo={}".format(plot.histosStack[0].GetName().split("__")[-1]))
        for index, sampleHisto in enumerate(plot.histos):
            histo = copy.deepcopy(sampleHisto)
            newHistos.append(copy.deepcopy(histo.ProjectionX(histo.GetName()+"projx", 1, 1)))  # convert to 1-D nominal hist
            # plot.histos.append(copy.deepcopy(histo))
        plot.histos = newHistos
    plot.keysStack = keysStack
    plot.keys = keys
    plot.addZUncBand = zUncBand
    plot.makeRatio = makeRatio
    plot.makeNSigma = makeNSigma
    plot.ytit = "N(Events)"
    plot.ylog = "no"
    plot.name = variableName
    plot.stackColorIndexes = stackColorIndexes
    plot.stackFillStyleIds = stackFillStyleIds
    plot.pdf_folder = "pdf_eejj/"
    plot.png_folder = "png_eejj/"
    plot.root_folder = "root_eejj/"
    plot.suffix = "eejj"
    plot.lumi_fb = ilumi

    return plot


def makeDefaultPlot2D(
    variableName, histoBaseName, samplesForStackHistos, sampleForDataHisto
):

    plot = Plot2D()
    plot.hasData = True
    plot.name = variableName
    plot.histosStack = generateHistoList(
        histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other], 1.0, True, "OTHERBKG"
    ) + generateHistoList(
        histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets]
    )
    plot.histodata = generateHisto(
        histoBaseName, sampleForDataHisto, variableName, dataMCFilesDict[sampleForDataHisto]
    )
    plot.suffix = "eejj"

    return plot


def makeDefaultPlot2D_NoData(
    variableName, histoBaseName, samplesForStackHistos, sampleForDataHisto
):

    plot = Plot2D()
    plot.hasData = False
    plot.name = variableName
    plot.histosStack = generateHistoList(
        histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other], 1.0, True, "OTHERBKG"
    ) + generateHistoList(
        histoBaseName, samplesForStackHistos_ttbar, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ttbar]
    ) + generateHistoList(
        histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets]
    )
    plot.histodata = generateHistoBlank(
        histoBaseName, sampleForDataHisto, variableName, dataMCFilesDict[sampleForDataHisto]
    )
    plot.keysStack = keysStack
    plot.keys = keys
    plot.pdf_folder = "pdf_eejj/"
    plot.png_folder = "png_eejj/"
    plot.root_folder = "root_eejj/"
    plot.suffix = "eejj"
    plot.lumi_fb = ilumi

    return plot


def makeDefaultPlot2D_NSigma(
    variableName, histoBaseName, samplesForStackHistos, sampleForDataHisto
):

    plot = Plot2DNSigma()
    plot.histosStack = generateHistoList(
        histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other], 1.0, True, "OTHERBKG"
    ) + generateHistoList(
        histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets]
    )
    plot.histodata = generateHisto(
        histoBaseName, sampleForDataHisto, variableName, dataMCFilesDict[sampleForDataHisto]
    )
    plot.gif_folder = "gif_eejj_scaled_finalOnly_2012A/"
    plot.eps_folder = "eps_eejj_scaled_finalOnly_2012A/"
    plot.suffix = "eejj_2DNSigma_finalOnly"

    return plot


def makeDefaultPlot2D_Ratio(
    variableName, histoBaseName, samplesForStackHistos, sampleForDataHisto
):

    plot = Plot2DRatio()
    plot.histosStack = generateHistoList(
        histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other], 1.0, True, "OTHERBKG"
    ) + generateHistoList(
        histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets]
    )
    plot.histodata = generateHisto(
        histoBaseName, sampleForDataHisto, variableName, dataMCFilesDict[sampleForDataHisto]
    )
    plot.gif_folder = "gif_eejj_scaled_finalOnly_2012A/"
    plot.eps_folder = "eps_eejj_scaled_finalOnly_2012A/"
    plot.suffix = "eejj_2DRatio_finalOnly"

    return plot


plots = []

####################################################################################################
#  PRESELECTION
####################################################################################################
if doPreselPlots:
    print("INFO: creating preselection plots...", end=' ')
    # plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # # plots[-1].ymax = 1e4
    # # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 200.0
    # plots[-1].xmax = 3000.0
    # # plots[-1].rebin = 20
    # plots[-1].rebin = 5
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    # plots[-1].xtitPaper = "S_{T} (GeV)"

    # plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].xmin = 200.0
    # plots[-1].xmax = 3000.0
    # plots[-1].rebin = 5
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    # plots[-1].xtitPaper = "S_{T} (GeV)"

    plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xmin = 200.0
    plots[-1].xmax = 3000.0
    plots[-1].rebin = "var"
    plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    plots[-1].ylog = "yes"
    plots[-1].name = "sT_PAS_rebinVar"
    plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T} (GeV)"
    plots[-1].lpos = "top-right"

    # plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 5
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    # plots[-1].xtitPaper = "m_{eejj} (GeV)"

    # plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 10
    # plots[-1].ylog = "yes"
    # plots[-1].name = "Meejj_PAS_rebin10"
    # plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    # plots[-1].xtitPaper = "m_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = "var"
    plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    plots[-1].ylog = "yes"
    plots[-1].name = "Meejj_PAS_rebinVar"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    plots[-1].xtitPaper = "m_{eejj} (GeV)"
    plots[-1].lpos = "top-right"

    # plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 5
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    # plots[-1].xtitPaper = "m_{eejj} (GeV)"

    # plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 10
    # plots[-1].ylog = "yes"
    # plots[-1].name = "Meejj_trainingSelection_rebin10"
    # plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    # plots[-1].xtitPaper = "m_{eejj} (GeV)"

    # plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = "var"
    # plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    # plots[-1].ylog = "yes"
    # plots[-1].name = "Meejj_trainingSelection_rebinVar"
    # plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    # plots[-1].xtitPaper = "m_{eejj} (GeV)"

    # plots.append(makeDefaultPlot("Mej_selected_min_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].ylog = "yes"
    # plots[-1].xmin = 0
    # plots[-1].rebin = 5
    # plots[-1].xtit = "M(ej) minimum (GeV) [Preselection]"
    # plots[-1].xtitPaper = "m_{ej}^{min} (GeV)"

    plots.append(makeDefaultPlot("Mej_selected_min_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = "var"
    plots[-1].xbins = [0, 20, 50, 90, 130, 180, 230, 280, 340, 410, 480, 550, 630, 720, 800, 900, 1000, 1110, 1220, 1330, 1450, 1580, 1710, 1840, 2000] #, 1980]
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    plots[-1].name = "Mej_selected_min_PAS_rebinVar"
    plots[-1].xtit = "M(ej) minimum (GeV) [Preselection]"
    plots[-1].xtitPaper = "m_{ej}^{min} (GeV)"
    plots[-1].ndivisions = 505
    plots[-1].lpos = "top-right"

    lq1500rescale = 50
    bdtRebin = 125
    # bdtBlindAbove = 0.8
    bdtBlindAbove = -1  # unblind
    bdtRebin = 200
    bdtYMax = 1e8
    lqmasstouse = 1500
    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ1500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegion_LQ1500_fixRebin"
    plots[-1].extraText = "m_{LQ} = 1500 GeV"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale
    plots[-1].histoColorIndexes = [4, 418, 1, 1, 4, 1]
    plots[-1].histoLineIndexes = [2, 1, 3, 1, 2, 3]
    plots[-1].lpos = "top-right"
    plots[-1].ratioYMax = 3

    print("Done")

####################################################################################################
#  FINAL SELECTION
####################################################################################################
if doFinalSelectionPlots:
    ymaxFinalSel = 1e8
    yminFinalSel = 1e-3
    mej_rebin = 2
    mej_xmin = 0
    mej_xmax = 3000
    mee_rebin = 1
    st_rebin = 2
    dr_rebin = 2
    ratioymax = 30
    print("INFO: creating final selection plots...", end=' ')
    for mass_point in LQmassesFinalSelection:
        samplesForHistos = [signalSampleName.format(mass_point)]
        keys = [signalSampleLabel.format(mass_point)]
        if blindFinalSelectionData:
            sampleForDataHisto = ""

        plots.append(
            makeDefaultPlot(
                "Mej_selected_min_LQ" + str(mass_point),
                histoBaseName_userDef,
                samplesForHistos,
                keys,
                samplesForStackHistos,
                keysStack,
                sampleForDataHisto,
                zUncBand,
                makeRatio,
                systs=doSystematics,
                rescaleDYJTTBarSystsAtPreselection=True if not options.preFit and not options.postFit else False,
                fitDiagFilePath=options.fitDiagFilepath if options.preFit or options.postFit else None,
                postFitJSON=options.postFitJSON if options.preFit or options.postFit else None,
                doPrefit=options.preFit,
                fitType=options.fitType,
                massPoint=int(mass_point)
            )
        )
        plots[-1].rebin = 4
        plots[-1].xtit = (
            "M_{ej}^{min} (GeV), (LQ M = " + str(mass_point) + " selection)"
        )
        plots[-1].xmin = mej_xmin
        plots[-1].xmax = mej_xmax
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "m_{ej}^{min} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymax = ymaxFinalSel
        plots[-1].ymin = yminFinalSel
        plots[-1].ymin = 1e-2
        plots[-1].ymax = 1e8
        if mass_point == 800:
            plots[-1].xmin = 200
            plots[-1].xmax = 3000
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04
        plots[-1].lpos = "top-right"
        plots[-1].ratioYMax = ratioymax

        plots.append(
            makeDefaultPlot(
                "Meejj_LQ" + str(mass_point),
                histoBaseName_userDef,
                samplesForHistos,
                keys,
                samplesForStackHistos,
                keysStack,
                sampleForDataHisto,
                zUncBand,
                makeRatio,
                systs=doSystematics,
                rescaleDYJTTBarSystsAtPreselection=True if not options.preFit and not options.postFit else False,
                fitDiagFilePath=options.fitDiagFilepath if options.preFit or options.postFit else None,
                postFitJSON=options.postFitJSON if options.preFit or options.postFit else None,
                doPrefit=options.preFit,
                fitType=options.fitType,
                massPoint=int(mass_point)
            )
        )
        plots[-1].xtit = (
            "M_{eejj} (GeV), (LQ M = " + str(mass_point) + " selection)"
        )
        # plots[-1].xmin = mej_xmin
        # plots[-1].xmax = mej_xmax
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "m_{eejj} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymax = ymaxFinalSel
        plots[-1].ymin = yminFinalSel
        if mass_point == 800:
            plots[-1].xmin = 1200
            plots[-1].xmax = 4000
        plots[-1].rebin = 20
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04
        plots[-1].lpos = "top-right"
        plots[-1].ratioYMax = ratioymax

        plots.append(
            makeDefaultPlot(
                "sT_eejj_LQ" + str(mass_point),
                histoBaseName_userDef,
                samplesForHistos,
                keys,
                samplesForStackHistos,
                keysStack,
                sampleForDataHisto,
                zUncBand,
                makeRatio,
                systs=doSystematics,
                rescaleDYJTTBarSystsAtPreselection=True if not options.preFit and not options.postFit else False,
                fitDiagFilePath=options.fitDiagFilepath if options.preFit or options.postFit else None,
                postFitJSON=options.postFitJSON if options.preFit or options.postFit else None,
                doPrefit=options.preFit,
                fitType=options.fitType,
                massPoint=int(mass_point)
            )
        )
        plots[-1].xtit = "S_{T} (GeV), (LQ M = " + str(mass_point) + " selection)"
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "S_{T} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymax = ymaxFinalSel
        plots[-1].ymin = yminFinalSel
        if mass_point == 800:
            plots[-1].xmin = 800
            plots[-1].xmax = 3000
        plots[-1].rebin = st_rebin
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04
        plots[-1].lpos = "top-right"
        plots[-1].ratioYMax = ratioymax

    print("Done")

####################################################################################################
# --- Generate and print the plots from the list 'plots' define above
####################################################################################################
CMS.setCMSStyle()

plotsPaper = copy.deepcopy(plots)

c = TCanvas()
fileps = "allPlots_eejj_paper.ps"
print("INFO: Doing paper plots")
print("INFO: writing canvas with plots to PDF...", end=' ')
c.Print(fileps + "[")
for i_plot, plot in enumerate(plotsPaper):
    try:
        plot.doPreliminary = False
        plot.Draw(fileps, i_plot + 1, style="paper")
    except Exception as e:
        raise RuntimeError("Caught exception while making plot", plot.name, ":", e)
c.Print(fileps + "]")
print("DONE")
print("INFO: MakeTOC()")
makeTOC("allPlots_eejj_paper_toc.tex", fileps, plotsPaper)

print("INFO: years =", qcdYearsToUse)
print("INFO: using file path:", dataMCFilePath)
if doQCD:
    print("INFO: using QCD file path:", inputFileQCDPath)

for tfile in dataMCFilesDict.values():
    tfile.Close()
