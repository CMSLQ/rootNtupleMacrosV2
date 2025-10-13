#!/usr/bin/env python3

import sys
import math
import copy
import numpy as np
import ctypes
from optparse import OptionParser
from plot_class import GetFile, Plot, Plot2D, generateHistoList, generateHisto, generateHistoBlank, generateHistoListFakeSystsFromModel, makeTOC, QuasiRebinHisto
from ROOT import gROOT, kCyan, kRed, TCanvas, TGraphAsymmErrors, TH1D, TFile
import cmsstyle as CMS


def SeparateArgs(arg):
    split = arg.split(",")
    return split


gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")
# gErrorIgnoreLevel = kWarning  # doesn't work

usage = "usage: %prog [options] \nExample: \npython makeStackHistoTemplateV2_eejj.py -q qcdFilePath -m mcDataFilePath -y years [-s signalName] [--fitDiagFilepath path] [--preFit] [--postFit] [--fitType postFitType]"
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
doPreselPlots = False
doBTagPlots = False
doFinalSelectionPlots = True
doFullSet = True
blindFinalSelectionData = False
doPrefit = options.preFit
doPostfit = options.postFit
if doPrefit or doPostfit:
    doPreselPlots = False
    print("INFO: Not doing preselection plots as prefit or postfit option is enabled.")
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
    elif '2016postVFP' in year:
        do2016 = True
        do2016post = True
        qcdYearsToUse = ['2016postVFP']
    elif '2016' in year:
        do2016 = True
        qcdYearsToUse = ['2016preVFP', '2016postVFP']
    elif '2017' in year:
        do2017 = True
        qcdYearsToUse = ['2017']
    elif '2018' in year:
        do2018 = True
        qcdYearsToUse = ['2018']
    elif 'RunII' == year or 'all' == year.lower():
        doRunII = True
        qcdYearsToUse = ['2016preVFP', '2016postVFP', '2017', '2018']
    else:
        print("ERROR: could not find one of 2017/2017/2018/RunII/all in given year. cannot do year-specific customizations. quitting.")
        exit(-1)

# LQmasses = [700, 1500]  # new samples don't have 650 GeV point
#LQmasses = [1500, 2000]  # new samples don't have 650 GeV point
LQmasses = [1000, 1200, 1500]
# LQmassesFinalSelection = [1200, 1500]
LQmassesFinalSelection = list(range(300, 3100, 100))
# LQmassesFinalSelection = [1200]

zUncBand = "no"
makeRatio = 1
makeNSigma = 1

pt_rebin = 2

if doSystematics:
    systNames = ["Prefire", "EES", "EER", "JES", "JER", "EleRecoSF", "EleIDSF", "EleTrigSF", "Pileup", "LHEPdf", "LHEScale"]
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
    # File_QCD_preselection = GetFile(
    #     inputFileQCD
    # )
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
    ilumi = "138.0"
    samplesForStackHistos_ZJets = ["ZJet_amcatnlo_ptBinned_IncStitch"]
    # samplesForStackHistos_other = ["SingleTop", "DIBOSON_nlo"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_other = ["OTHERBKG_dibosonNLO_singleTop"]  # UL with single/double QCD fakes from data
    samplesForStackHistos_ttbar = ["TTTo2L2Nu"]  # UL with single/double QCD fakes from data
    keysStack = ["QCD multijet (data)"] if doQCD else []
    keysStack.extend([
        "Other backgrounds",
        # "t#bar{t} (powheg)",
        # "Z/#gamma* + jets (MG5_aMC Pt+IncStitch)",
        "t#bar{t}",
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
samplesForHistos = list(set(samplesForHistos))
# samplesForHistos = list(samplesForHistos)
extraLQMasses = [500]
extraSamplesForHistos = [signalSampleName.format(lqmass) for lqmass in extraLQMasses]
for sample in samplesForHistos + extraSamplesForHistos:
    dataMCFilesDict[sample] = GetFile(dataMCFilePath + "analysisClass_lq_eejj_{}_plots.root".format(sample))
keys = [signalSampleLabel.format(lqmass) for lqmass in LQmasses+extraLQMasses]
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
    doPrefit=False,
    fitType=None
):
    plot = Plot()
    systList = []
    if systs:
        histoBaseName = histoBaseName.replace("histo1D", "histo2D")
        histoBaseName+="WithSystematics"
        systList = systNames
    plot.systNames = systList
    plot.histosStack = []
    plot.histosStack.extend(generateHistoList(
            histoBaseName, samplesForStackHistos_other, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_other], 1.0, True, histoBaseName.replace("SAMPLE", "OTHERBKG").replace("VARIABLE", variableName), 
            years=qcdYearsToUse,
            masses=LQmassesFinalSelection,
            fitDiagFilePath=fitDiagFilePath,
            doPrefit=doPrefit,
            fitType=fitType
        ))
    plot.histosStack.extend(generateHistoList(
            histoBaseName, samplesForStackHistos_ttbar, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ttbar], rescaleSystsAtPreselection=rescaleDYJTTBarSystsAtPreselection,
            normSyst=ttbarNormSyst,
            years=qcdYearsToUse,
            masses=LQmassesFinalSelection,
            fitDiagFilePath=fitDiagFilePath,
            doPrefit=doPrefit,
            fitType=fitType
            
        ))
    plot.histosStack.extend(generateHistoList(
        histoBaseName, samplesForStackHistos_ZJets, variableName, [dataMCFilesDict[sample] for sample in samplesForStackHistos_ZJets], rescaleSystsAtPreselection=rescaleDYJTTBarSystsAtPreselection,
        normSyst=dyjNormSyst,
        years=qcdYearsToUse,
            masses=LQmassesFinalSelection,
        fitDiagFilePath=fitDiagFilePath,
        doPrefit=doPrefit,
        fitType=fitType
        ))
    if systs:
        qcdHists = []
        for year in qcdYearsToUse:
            qcdFile = TFile.Open(inputFileQCDPath.format(year))
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
                fitDiagFilePath=options.fitDiagFilepath if options.preFit or options.postFit else None,
                doPrefit=options.preFit,
                fitType=options.fitType
            ) if doQCD else []
            if len(qcdHists) < 1:
                qcdHists = qcdHistsThisYear
            else:
                for idx, hist in enumerate(qcdHists):
                    hist.Add(qcdHistsThisYear[idx])
            qcdFile.Close()
    else:
        qcdHists = generateHistoList(
            histoBaseName,
            samplesForStackHistos_QCD,
            variableName,
            File_QCD_preselection
        ) if doQCD else []
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
        histoBaseName, samplesForHistos, variableName, [dataMCFilesDict[sample] for sample in samplesForHistos]
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
        #print("INFO: redoing stack plots for histo={}".format(plot.histosStack[0].GetName().split("__")[-1]))
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
    plots.append(makeDefaultPlot("nElectron_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].ymax = 10000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -0.5
    plots[-1].xmax = 6.5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Number of electrons [Preselection]"
    plots[-1].xtitPaper = "N(e)"

    plots.append(makeDefaultPlot("nMuon_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].ymax = 10000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -0.5
    plots[-1].xmax = 6.5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Number of muons [Preselection]"
    plots[-1].xtitPaper = "N(#mu)"

    plots.append(makeDefaultPlot("nJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xtit = "Number of jets [Preselection]"
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = -0.5
    plots[-1].xmax = 10.5
    plots[-1].xtitPaper = "N(jet)"

    plots.append(makeDefaultPlot("nJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xtit = "Number of jets [Preselection]"
    # plots[-1].ymax = 5e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xmin = -0.5
    plots[-1].xmax = 10.5
    plots[-1].xtitPaper = "N(jet)"

    # plots.append(makeDefaultPlot("EleChargeSum_PAS"))
    # plots[-1].xtit  = "Electron 1 charge + Electron 2 charge [Preselection]"
    # plots[-1].ymax  = 200000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xmin  = -1.5
    # plots[-1].xmax  = 1.5
    #
    #
    #
    # plots.append(makeDefaultPlot("EleChargeSum_PAS"))
    # plots[-1].xtit  = "Electron 1 charge + Electron 2 charge [Preselection]"
    # plots[-1].ymax  = 20000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -1.5
    # plots[-1].xmax  = 1.5

    # plots.append(makeDefaultPlot("PtHeep1stEle_PAS"))
    plots.append(makeDefaultPlot("Pt1stEle_PAS", systs=doSystematics, dataBlindAbove=dataBlindAbovePt1, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 50
    plots[-1].xmax = 1500
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Electron p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(e1) (GeV)"

    # plots.append(makeDefaultPlot("PtHeep2ndEle_PAS"))
    plots.append(makeDefaultPlot("Pt2ndEle_PAS", systs=doSystematics, dataBlindAbove=dataBlindAbovePt2, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 1e5
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 50
    plots[-1].xmax = 1000
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Electron p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(e2) (GeV)"

    plots.append(makeDefaultPlot("Pt1stMuon_PAS", dataBlindAbove=dataBlindAbovePt1))
    # plots[-1].rebin = 2
    # plots[-1].ymax = 1e5
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 200
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Muon p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(#mu 1) (GeV)"

    plots.append(makeDefaultPlot("Pt2ndMuon_PAS", dataBlindAbove=dataBlindAbovePt2))
    # plots[-1].rebin = 2
    # plots[-1].ymax = 1e5
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 200
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Muon p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(#mu 2) (GeV)"

    plots.append(makeDefaultPlot("Eta1stEle_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Electron #eta [Preselection]"
    plots[-1].xtitPaper = "#eta(e1) (GeV)"

    plots.append(makeDefaultPlot("Phi1stEle_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    # plots[-1].ymax = 5e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Electron #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(e1) (GeV)"

    plots.append(makeDefaultPlot("Eta2ndEle_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].rebin = 2
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Electron #eta [Preselection]"
    plots[-1].xtitPaper = "#eta(e2) (GeV)"

    plots.append(makeDefaultPlot("Phi2ndEle_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    # plots[-1].ymax = 5e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Electron #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(e1) (GeV)"

    plots.append(makeDefaultPlot("Charge1stEle_PAS"))
    # plots[-1].ymin = 0.0
    # plots[-1].ymax = 8e4
    plots[-1].xtit = "1st Electron Charge [Preselection]"
    plots[-1].xtitPaper = "Charge(e1)"

    plots.append(makeDefaultPlot("Charge2ndEle_PAS"))
    # plots[-1].ymin = 0.0
    # plots[-1].ymax = 8e4
    plots[-1].xtit = "2nd Electron Charge [Preselection]"
    plots[-1].xtitPaper = "Charge(e2)"

    # plots.append(makeDefaultPlot("MTenu_PAS"))
    # plots[-1].xtit = "M_{T}(e_{1}m PFMET [Preselection]) (GeV)"
    # plots[-1].rebin = 1
    # # plots[-1].ymax = 300000
    # # plots[-1].ymin = 1e-1
    # plots[-1].xmax = 500
    # plots[-1].xmin = 0
    # plots[-1].rebin = 2
    # plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("MET_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 800
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "PFMET (GeV) [Preselection]"
    plots[-1].xtitPaper = "MET (GeV)"

    plots.append(makeDefaultPlot("METPhi_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    # plots[-1].ymax = 1.2e4
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    plots[-1].xtit = "PFMET #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(MET)"

    plots.append(makeDefaultPlot("MET_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xtit = "PFMET (GeV) [Preselection]"
    plots[-1].rebin = 4
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 500
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "MET (GeV)"

    plots.append(makeDefaultPlot("Pt1stJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = pt_rebin
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 2000
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Jet p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(j1) (GeV)"

    plots.append(makeDefaultPlot("Pt2ndJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = pt_rebin
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 2000
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(j2) (GeV)"

    plots.append(makeDefaultPlot("Pt3rdJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = pt_rebin
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 2000
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "3rd Jet p_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "p_{T}(j2) (GeV)"

    plots.append(makeDefaultPlot("Eta1stJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Jet #eta [Preselection]"
    plots[-1].xtitPaper = "#eta(j1)"

    plots.append(makeDefaultPlot("Eta2ndJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet #eta [Preselection]"
    plots[-1].xtitPaper = "#eta(j2)"

    plots.append(makeDefaultPlot("Eta3rdJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "3rd Jet #eta [Preselection]"
    plots[-1].xtitPaper = "#eta(j3)"

    plots.append(makeDefaultPlot("Phi1stJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Jet #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(j1)"

    plots.append(makeDefaultPlot("Phi2ndJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(j2)"

    plots.append(makeDefaultPlot("Phi3rdJet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "3rd Jet #phi [Preselection]"
    plots[-1].xtitPaper = "#phi(j3)"

    plots.append(makeDefaultPlot("sTlep_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 3000
    plots[-1].xmin = 200
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (1st Electron, 2nd Electron) (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T}(e1, e2) (GeV)"

    plots.append(makeDefaultPlot("sTjet_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 200
    plots[-1].xmax = 3000
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (1st Jet, 2nd Jet) (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T}(j1, j2) (GeV)"

    plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 200.0
    plots[-1].xmax = 3000.0
    # plots[-1].rebin = 20
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T} (GeV)"

    plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xmin = 200.0
    plots[-1].xmax = 3000.0
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T} (GeV)"

    plots.append(makeDefaultPlot("sT_PAS", systs=doSystematics, dataBlindAbove=dataBlindAboveSt, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].xmin = 200.0
    plots[-1].xmax = 3000.0
    plots[-1].rebin = "var"
    plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    plots[-1].ylog = "yes"
    plots[-1].name = "sT_PAS_rebinVar"
    plots[-1].xtit = "S_{T} (GeV) [Preselection]"
    plots[-1].xtitPaper = "S_{T} (GeV)"

    # plots.append(makeDefaultPlot("sT_zjj_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 200.
    # plots[-1].xmax = 2000.
    # plots[-1].rebin = 2
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "S_{T} (Z, 2jet) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet1_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jet 1 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet2_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jet 2 (GeV) [Preselection]"
    #
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Ele1_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from ele 1 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlo("sTfrac_Ele2_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from ele 2 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Ele_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from electrons (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet_PAS"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jets (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Mjj_PAS"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Dijet Mass (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{jj} (GeV)"

    # plots.append(makeDefaultPlot("M_j2j3_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(j2,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_j1j3_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(j1,j3) Mass (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Me1j1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 5e4
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e1,j1) Mass (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e1,j1} (GeV)"

    plots.append(makeDefaultPlot("Me1j1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 9.5e3
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].xtit = "M(e1,j1) Mass (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e1,j1} (GeV)"

    # plots.append(makeDefaultPlot("M_e1j3_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e1,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_e2j3_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e2,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_eejjj_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e1, e2, j1, j2, j3) (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 10
    plots[-1].ylog = "yes"
    plots[-1].name = "Meejj_PAS_rebin10"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = "var"
    plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    plots[-1].ylog = "yes"
    plots[-1].name = "Meejj_PAS_rebinVar"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 10
    plots[-1].ylog = "yes"
    plots[-1].name = "Meejj_trainingSelection_rebin10"
    plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Meejj_trainingSelection", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = "var"
    plots[-1].xbins = [300, 330, 370, 420, 480, 550, 630, 720, 820, 930, 1050, 1180, 1320, 1470, 1630, 1800, 1980, 2170, 2370, 2580, 2800, 3000]
    plots[-1].ylog = "yes"
    plots[-1].name = "Meejj_trainingSelection_rebinVar"
    plots[-1].xtit = "Mass_{eejj} (GeV) [training]"
    plots[-1].xtitPaper = "M_{eejj} (GeV)"

    plots.append(makeDefaultPlot("Mee_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    plots[-1].rebin = "var"
    plots[-1].xbins = list(range(0, 410, 10)) + [
        420,
        440,
        460,
        480,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
    ]
    plots[-1].xmin = 0.0
    plots[-1].xmax = 2000.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg.]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].addOvfl = "no"
    plots[-1].ymin = 1e-1
    plots[-1].ymax = 1e10
    #plots[-1].rebin = "var"
    #plots[-1].xbins = list(range(0, 410, 10)) + [
    #    420,
    #    440,
    #    460,
    #    480,
    #    500,
    #    550,
    #    600,
    #    650,
    #    700,
    #    800,
    #    900,
    #    1000,
    #    1200,
    #    1400,
    #    1600,
    #    1800,
    #    2000,
    #]
    plots[-1].xmin = 70.0
    plots[-1].xmax = 110.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg., 70-110 GeV]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_BkgControlRegion_gteTwoBtaggedJets", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    #plots[-1].rebin = 1
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].addOvfl = "no"
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg, >= 2 b-tags [GeV]]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    plots[-1].rebin = "var"
    plots[-1].xbins = list(range(0, 410, 10)) + [
        420,
        440,
        460,
        480,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
    ]
    # plots[-1].ymax = 2e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0.0
    plots[-1].xmax = 2000.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    #plots[-1].rebin = "var"
    #plots[-1].xbins = range(0, 410, 10) + [
    #    420,
    #    440,
    #    460,
    #    480,
    #    500,
    #    550,
    #    600,
    #    650,
    #    700,
    #    800,
    #    900,
    #    1000,
    #    1200,
    #    1400,
    #    1600,
    #    1800,
    #    2000,
    #]
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 100.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].name = "Mee_100_250_PAS"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, 100-250 GeV]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].name = "Mee_60_250_PAS"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, 60-250 GeV]"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_EBEB_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EB-EB) (GeV)"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_EBEE_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2e3
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EB-EE) (GeV)"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_EEEE_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    #plots[-1].ymax = 1e3
    #plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EE-EE) (GeV)"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_End2End2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    #plots[-1].ymax = 1e3
    #plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EE2-EE2) (GeV)"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    plots.append(makeDefaultPlot("Mee_EB_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 8e3
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0.0
    plots[-1].xmax = 1000.0
    plots[-1].xtit = "M(ee) (preselection, EB-EE and EB-EB) (GeV)"
    plots[-1].xtitPaper = "M_{ee} (GeV)"

    # plots.append(makeDefaultPlot("Mee_sT300To500_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # plots[-1].addOvfl = "no"
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 1e6
    # plots[-1].xmin = 70.0
    # plots[-1].xmax = 110.0
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg., sT 300 - 500 GeV]"

    # plots.append(makeDefaultPlot("Mee_sT500To750_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # plots[-1].addOvfl = "no"
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 1e6
    # plots[-1].xmin = 70.0
    # plots[-1].xmax = 110.0
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg., sT 500 - 750 GeV]"

    # plots.append(makeDefaultPlot("Mee_sT750To1250_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # plots[-1].addOvfl = "no"
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 1e6
    # plots[-1].xmin = 70.0
    # plots[-1].xmax = 110.0
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg., sT 750 - 1250 GeV]"

    # plots.append(makeDefaultPlot("Mee_sT1250ToInf_BkgControlRegion", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # plots[-1].addOvfl = "no"
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 1e6
    # plots[-1].xmin = 70.0
    # plots[-1].xmax = 110.0
    # plots[-1].ylog = "yes"
    # plots[-1].xtit = "M(ee) (GeV) [Bkg. Ctrl. Reg., sT > 1250 GeV]"

    #plots.append(makeDefaultPlot("Mee_EBEB_80_100_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 8e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EB) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EBEE_80_100_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1500
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EE) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EEEE_80_100_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EE-EE) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EB_80_100_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 6e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EE and EB-EB) (GeV)"

    #plots.append(makeDefaultPlot("Mee_80_100_Preselection"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e8
    ## plots[-1].ymin = 1e-1
    #plots[-1].rebin = 4
    #plots[-1].xmin = 70.0
    #plots[-1].xmax = 110.0
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "M(ee) [80, 100] (GeV) [Preselection]"

    #plots.append(makeDefaultPlot("Mee_EBEB_70_110_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e4
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EB) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EBEE_70_110_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EE) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EEEE_70_110_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EE-EE) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EB_70_110_PAS"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 6e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EE and EB-EB) (GeV)"

    #plots.append(makeDefaultPlot("Mee_70_110_Preselection"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e8
    ## plots[-1].ymin = 1e-1
    #plots[-1].rebin = 4
    #plots[-1].xmin = 70.0
    #plots[-1].xmax = 110.0
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "M(ee) [70, 110] (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Ptee_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 2
    # plots[-1].ymax = 4e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 1000
    plots[-1].ylog = "yes"
    plots[-1].xtit = "P_{T}(ee) (GeV) [Preselection]"
    plots[-1].xtitPaper = "P_{T}(ee) (GeV)"

    #plots.append(makeDefaultPlot("Mee_70_110_Preselection"))
    #plots[-1].rebin = 4
    ## plots[-1].ymax = 2e8
    ## plots[-1].ymin = 1e-1
    #plots[-1].rebin = 8
    #plots[-1].xmin = 70.0
    #plots[-1].xmax = 110.0
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "M(ee) [70, 110] (GeV) [Preselection]"

    # plots.append(makeDefaultPlot("Ptj1j2_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(j_{1}, j_{2}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj2j3_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{2}, j_{3}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj1j3_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{1}, j_{3}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj1j2j3_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{1}, jet_{2}, j_{3}) (GeV) [Preselection]"
    #
    #
    #
    # plots.append(makeDefaultPlot("Ptee_Minus_Ptj1j2_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -500
    # plots[-1].xmax  = 500
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(ee) - P_{T}(j_{1}, j_{2}) (GeV) [Preselection]"

    # plots.append(makeDefaultPlot("Ptee_Minus_Ptj1j2j3_PAS"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -500
    # plots[-1].xmax  = 500
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(ee) - P_{T}(j_{1}, j_{2}, j_{3}) (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Me1j1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{1}j_{1}) (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e_{1}j_{1}} (GeV)"

    plots.append(makeDefaultPlot("Me1j2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{1}j_{2}) (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e_{1}j_{2}} (GeV)"

    plots.append(makeDefaultPlot("Me2j1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{2}j_{1}) (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e_{2}j_{1}} (GeV)"

    plots.append(makeDefaultPlot("Me2j2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].xtit = "M(e_{2}j_{2}) (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{e_{2}j_{2}} (GeV)"

    # plots.append(makeDefaultPlot("Mej_selected_avg_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # # plots[-1].ymax = 1.8e4
    # # plots[-1].ymin = 1e-1
    # plots[-1].ylog = "no"
    # plots[-1].xmin = 0
    # # plots[-1].xmax = 1000
    # plots[-1].rebin = 5
    # plots[-1].xtit = "M(ej) average (GeV) [Preselection]"
    # plots[-1].xtitPaper = "M_{ej}^{avg} (GeV)"

    # USED IN AN
    plots.append(makeDefaultPlot("Mej_selected_avg_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1200
    plots[-1].rebin = 5
    plots[-1].xtit = "M(ej) average (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{avg} (GeV)"

    # plots.append(makeDefaultPlot("Mej_selected_min_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    # plots[-1].rebin = 1
    # # plots[-1].ymax = 1.8e4
    # # plots[-1].ymin = 1e-1
    # plots[-1].ylog = "no"
    # plots[-1].xmin = 0
    # # plots[-1].xmax = 1000
    # plots[-1].rebin = 5
    # plots[-1].xtit = "M(ej) minimum (GeV) [Preselection]"
    # plots[-1].xtitPaper = "M_{ej}^{min} (GeV)"

    plots.append(makeDefaultPlot("Mej_selected_min_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    plots[-1].rebin = 5
    plots[-1].xtit = "M(ej) minimum (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{min} (GeV)"

    plots.append(makeDefaultPlot("Mej_selected_min_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = "var"
    plots[-1].xbins = [0, 20, 50, 90, 130, 180, 230, 280, 340, 410, 480, 550, 630, 720, 800, 900, 1000, 1110, 1220, 1330, 1450, 1580, 1710, 1840, 1980]
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    plots[-1].name = "Mej_selected_min_PAS_rebinVar"
    plots[-1].xtit = "M(ej) minimum (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{min} (GeV)"

    plots.append(makeDefaultPlot("Mej_selected_max_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1.8e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1000
    plots[-1].rebin = 5
    plots[-1].xtit = "M(ej) maximum (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{max} (GeV)"

    plots.append(makeDefaultPlot("Mej_selected_max_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1200
    plots[-1].rebin = 5
    plots[-1].xtit = "M(ej) maximum (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{max} (GeV)"

    # plots.append(makeDefaultPlot("Mej_selected_diff_PAS")
    # plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(ej) diff (GeV) [Preselection]"
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 500
    # plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_asym_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    #plots[-1].xmin = 0
    #plots[-1].xmax = 1200
    #plots[-1].rebin = 5
    plots[-1].xtit = "M(ej) asym (GeV) [Preselection]"
    plots[-1].xtitPaper = "M_{ej}^{asym} (GeV)"

    plots.append(makeDefaultPlot("nVertex_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].xmin = -0.5
    plots[-1].xmax = 60.5
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 5e7
    # plots[-1].ylog = "yes"
    plots[-1].xtit = "n(vertices) [Preselection]"
    plots[-1].xtitPaper = "N(vertex)"

    # plots.append(makeDefaultPlot("nVertex_PAS"))
    # plots[-1].rebin = 1
    # plots[-1].xmin = -0.5
    # plots[-1].xmax = 40.5
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 6e2
    # plots[-1].xtit = "n(vertices) [Preselection]"

    plots.append(makeDefaultPlot("DR_Ele1Jet1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].ylog  = "yes"
    plots[-1].xtit = "#DeltaR(e_{1},j_{1})"
    plots[-1].xtitPaper = "#DeltaR(e_{1},j_{1})"
    
    plots.append(makeDefaultPlot("DR_Ele1Jet2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].xtit = "#DeltaR(e_{1},j_{2})"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].ylog  = "yes"
    plots[-1].xtitPaper = "#DeltaR(e_{1},j_{2})"
    
    plots.append(makeDefaultPlot("DR_Ele2Jet1_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].xtit = "#DeltaR(e_{2},j_{1})"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].ylog  = "yes"
    plots[-1].xtitPaper = "#DeltaR(e_{2},j_{1})"
    
    plots.append(makeDefaultPlot("DR_Ele2Jet2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 1
    plots[-1].xtit = "#DeltaR(e_{2},j_{2})"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].ylog  = "yes"
    plots[-1].xtitPaper = "#DeltaR(e_{2},j_{2})"
    
    plots.append(makeDefaultPlot("DR_Ele1Ele2_PAS", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True))
    plots[-1].rebin = 4
    plots[-1].xtit = "#DeltaR(e_{1},e_{2})"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].ylog  = "yes"
    plots[-1].xtitPaper = "#DeltaR(e_{1},e_{2})"
    
    plots.append(makeDefaultPlot("minDR_EleJet_PAS"))
    plots[-1].rebin = 1
    plots[-1].xtit = "Minimum #DeltaR(e_{1},(j_{1}, j_{2}, j_{3}))"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].ylog  = "yes"
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].xtitPaper = "min(#DeltaR(e_{1},(j_{1}, j_{2}, j_{3})))"

    plots.append(makeDefaultPlot("minDR_ZJet_PAS"))
    plots[-1].rebin = 4
    plots[-1].xtit = "Minimum #DeltaR(ee, (j_{1}, j_{2})) [Preselection]"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].ylog  = "yes"
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].xtitPaper = "min(#DeltaR(ee, (j_{1}, j_{2}))) [Preselection]"
    
    plots.append(makeDefaultPlot("DR_ZJet1_PAS"))
    plots[-1].rebin = 4
    plots[-1].xtit = "#DeltaR(ee, j_{1}) [Preselection]"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].ylog  = "yes"
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].xtitPaper = "#DeltaR(ee, j_{1})"
    
    plots.append(makeDefaultPlot("DR_ZJet2_PAS"))
    plots[-1].rebin = 4
    plots[-1].xtit = "#DeltaR(ee, j_{2}) [Preselection]"
    plots[-1].ymax = 2000000
    plots[-1].ymin = 1e-1
    plots[-1].ylog  = "yes"
    plots[-1].xmin = 0
    plots[-1].xmax = 6
    plots[-1].xtitPaper = "#DeltaR(ee, j_{2})"

    #plots.append(makeDefaultPlot("Meej_PAS",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 10
    # plots[-1].ymax = 200000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Mass_{eej} (GeV) [Preselection]"

    #plots.append(makeDefaultPlot("Mejj_PAS",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 10
    # plots[-1].ymax = 200000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Mass_{ejj} (GeV) [Preselection]"

    #plots.append(makeDefaultPlot2D_NoData("MeeVsST_PAS",histoBaseName2D_userDef,samplesForStackHistos,sampleForDataHisto))
    # plots[-1].xrebin = 2
    # plots[-1].yrebin = 2
    # plots[-1].xtit = "M(ee) [GeV]"
    # plots[-1].ytit = "S_{T}(eejj) [GeV]"
    # plots[-1].zlog = "yes"
    #
    #plots.append(makeDefaultPlot2D_NoData("MeeVsST_PAS",histoBaseName2D_userDef,samplesForStackHistos,sampleForDataHisto))
    # plots[-1].xrebin = 2
    # plots[-1].yrebin = 2
    # plots[-1].xtit = "M(ee) [GeV]"
    # plots[-1].ytit = "S_{T}(eejj) [GeV]"
    #
    #
    lq1500rescale = 50
    lqmasstouse = 500
    plots.append(makeDefaultPlot("BDTOutput_ZJetCRRegion_LQ500", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [ZJetCRRegion, M_{LQ} = 500 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1000
    plots.append(makeDefaultPlot("BDTOutput_ZJetCRRegion_LQ1000", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [ZJetCRRegion, M_{LQ} = 1000 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1500
    plots.append(makeDefaultPlot("BDTOutput_ZJetCRRegion_LQ1500", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [ZJetCRRegion, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale

    lqmasstouse = 500
    plots.append(makeDefaultPlot("BDTOutput_TTBarCRRegion_LQ500", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [TTBarCRRegion, M_{LQ} = 500 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1000
    plots.append(makeDefaultPlot("BDTOutput_TTBarCRRegion_LQ1000", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [TTBarCRRegion, M_{LQ} = 1000 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1500
    plots.append(makeDefaultPlot("BDTOutput_TTBarCRRegion_LQ1500", systs=doSystematics, rescaleDYJTTBarSystsAtPreselection=True, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [TTBarCRRegion, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = 10
    plots[-1].ymax = 1e6
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale

    # bdtRebin = 125
    bdtBlindAbove = 0.8
    bdtRebin = 200
    bdtYMax = 1e8
    lqmasstouse = 500
    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 500 GeV]"
    plots[-1].rebin = 10
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 500 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegion_LQ500_fixRebin"
    plots[-1].extraText = "m_{LQ} = 500 GeV"
    plots[-1].xtitPaper = "BDT score"

    plots.append(makeDefaultPlot("BDTOutput_TrainRegionNoMeejj_LQ500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, no M_{eejj}, M_{LQ} = 500 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegionNoMeejj_LQ500_fixRebin"
    plots[-1].extraText = "m_{LQ} = 500 GeV"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1000
    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ1000", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 1000 GeV]"
    plots[-1].rebin = 10
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"

    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ1000", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 1000 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegion_LQ1000_fixRebin"
    plots[-1].extraText = "m_{LQ} = 1000 GeV"
    plots[-1].xtitPaper = "BDT score"

    plots.append(makeDefaultPlot("BDTOutput_TrainRegionNoMeejj_LQ1000", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, no M_{eejj}, M_{LQ} = 1000 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegionNoMeejj_LQ1000_fixRebin"
    plots[-1].extraText = "m_{LQ} = 1000 GeV"
    plots[-1].xtitPaper = "BDT score"

    lqmasstouse = 1500
    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ1500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = 10
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale

    plots.append(makeDefaultPlot("BDTOutput_TrainRegion_LQ1500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegion_LQ1500_fixRebin"
    plots[-1].extraText = "m_{LQ} = 1500 GeV"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale

    plots.append(makeDefaultPlot("BDTOutput_TrainRegionNoMeejj_LQ1500", dataBlindAbove=bdtBlindAbove, systs=doSystematics, samplesForHistos=[signalSampleName.format(lqmasstouse)], keys=[signalSampleLabel.format(lqmasstouse)]))
    plots[-1].xtit = "BDT output [training, no M_{eejj}, M_{LQ} = 1500 GeV]"
    plots[-1].rebin = bdtRebin
    plots[-1].ymax = bdtYMax
    plots[-1].ymin = 1e-1
    plots[-1].xmax = 1
    plots[-1].xmin = -1
    plots[-1].ylog = "yes"
    plots[-1].name = "BDTOutput_TrainRegionNoMeejj_LQ1500_fixRebin"
    plots[-1].extraText = "m_{LQ} = 1500 GeV"
    plots[-1].xtitPaper = "BDT score"
    plots[-1].histoRescaleFactor = lq1500rescale

    plots.append(makeDefaultPlot("EleTriggerMatches_PAS"))
    plots[-1].xtit = "Electron trigger object matching [preselection]"
    plots[-1].ylog = "yes"
    plots[-1].lpos = "top-left"
    plots[-1].ymax = 1e10
    plots[-1].ymin = 1e-1
    plots[-1].xtitPaper = "Electron trigger object matching"

    if doSystematics:
        plots.append(makeDefaultPlot2D_NoData("systematics",histoBaseName2D_userDef,samplesForStackHistos,sampleForDataHisto))
        #plots[-1].xtit = "M(ee) [GeV]"
        #plots[-1].ytit = "S_{T}(eejj) [GeV]"
        #plots[-1].zlog = "yes"

    print("Done")

####################################################################################################
#  PRESELECTION -- two B-tagged jets
####################################################################################################
if doBTagPlots:
    print("INFO: creating B-tag plots...", end=' ')
    plots.append(makeDefaultPlot("Mee_PAS_gteTwoBtaggedJets"))
    plots[-1].rebin = 4
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, >= 2 b-tags, GeV]"
    plots[-1].name = "Mee_PAS_gteTwoBtaggedJets"

    plots.append(makeDefaultPlot("Mee_PAS_gteOneBtaggedJet"))
    plots[-1].rebin = 4
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, >= 1 b-tag, GeV]"
    plots[-1].name = "Mee_PAS_gteOneBtaggedJet"

    plots.append(makeDefaultPlot("Mee_PAS_noBtaggedJets"))
    plots[-1].rebin = 4
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, == 0 b-tags, GeV]"
    plots[-1].name = "Mee_PAS_noBtaggedJets"

    plots.append(makeDefaultPlot("nElectron_gteTwoBtaggedJets"))
    # plots[-1].ymax = 10000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -0.5
    plots[-1].xmax = 6.5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Number of electrons [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("nMuon_gteTwoBtaggedJets"))
    # plots[-1].ymax = 10000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -0.5
    plots[-1].xmax = 6.5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Number of muons [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("nJet_gteTwoBtaggedJets"))
    plots[-1].xtit = "Number of jets [Preselection, >= 2 b-tags]"
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = -0.5
    plots[-1].xmax = 10.5

    plots.append(makeDefaultPlot("nJet_gteTwoBtaggedJets"))
    plots[-1].xtit = "Number of jets [Preselection, >= 2 b-tags]"
    # plots[-1].ymax = 5e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xmin = -0.5
    plots[-1].xmax = 10.5

    # plots.append(makeDefaultPlot("EleChargeSum_gteTwoBtaggedJets"))
    # plots[-1].xtit  = "Electron 1 charge + Electron 2 charge [Preselection]"
    # plots[-1].ymax  = 200000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xmin  = -1.5
    # plots[-1].xmax  = 1.5
    #
    #
    #
    # plots.append(makeDefaultPlot("EleChargeSum_gteTwoBtaggedJets"))
    # plots[-1].xtit  = "Electron 1 charge + Electron 2 charge [Preselection]"
    # plots[-1].ymax  = 20000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -1.5
    # plots[-1].xmax  = 1.5

    # plots.append(makeDefaultPlot("PtHeep1stEle_gteTwoBtaggedJets"))
    plots.append(makeDefaultPlot("Pt1stEle_gteTwoBtaggedJets", dataBlindAbove=dataBlindAbovePt1))
    plots[-1].rebin = 2
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 1500
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Electron p_{T} (GeV) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("PtHeep2ndEle_gteTwoBtaggedJets"))
    plots.append(makeDefaultPlot("Pt2ndEle_gteTwoBtaggedJets", dataBlindAbove=dataBlindAbovePt2))
    plots[-1].rebin = 2
    # plots[-1].ymax = 1e5
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 1000
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Electron p_{T} (GeV) [Preselection, >= 2 b-tags]"

    #plots.append(makeDefaultPlot("Pt1stMuon_gteTwoBtaggedJets", dataBlindAbove=dataBlindAbovePt1))
    ## plots[-1].rebin = 2
    ## plots[-1].ymax = 1e5
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0
    #plots[-1].xmax = 200
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "1st Muon p_{T} (GeV) [Preselection]"

    #plots.append(makeDefaultPlot("Pt2ndMuon_gteTwoBtaggedJets", dataBlindAbove=dataBlindAbovePt2))
    ## plots[-1].rebin = 2
    ## plots[-1].ymax = 1e5
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0
    #plots[-1].xmax = 200
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "2nd Muon p_{T} (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Eta1stEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "1st Electron #eta [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("Phi1stEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "1st Electron #phi [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 4
    # plots[-1].ymax = 5e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("Eta2ndEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "2nd Electron #eta [Preselection, >= 2 b-tags]"
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].rebin = 2
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("Phi2ndEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "2nd Electron #phi [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 4
    # plots[-1].ymax = 5e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("Charge1stEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "1st Electron Charge [Preselection, >= 2 b-tags]"
    # plots[-1].ymin = 0.0
    # plots[-1].ymax = 8e4

    plots.append(makeDefaultPlot("Charge2ndEle_gteTwoBtaggedJets"))
    plots[-1].xtit = "2nd Electron Charge [Preselection, >= 2 b-tags]"
    # plots[-1].ymin = 0.0
    # plots[-1].ymax = 8e4

    plots.append(makeDefaultPlot("MTenu_gteTwoBtaggedJets"))
    plots[-1].xtit = "M_{T}(e_{1}m PFMET [Preselection, >= 2 b-tags]) (GeV)"
    plots[-1].rebin = 1
    # plots[-1].ymax = 300000
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 500
    plots[-1].xmin = 0
    plots[-1].rebin = 2
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("MET_gteTwoBtaggedJets"))
    plots[-1].xtit = "PFMET (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 800
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("METPhi_gteTwoBtaggedJets"))
    plots[-1].xtit = "PFMET #phi [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 4
    # plots[-1].ymax = 1.2e4
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"

    plots.append(makeDefaultPlot("Pt1stJet_gteTwoBtaggedJets"))
    plots[-1].xtit = "1st Jet p_{T} (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].rebin = pt_rebin
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 2000
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"

    plots.append(makeDefaultPlot("Pt2ndJet_gteTwoBtaggedJets"))
    plots[-1].rebin = pt_rebin
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 2000
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet p_{T} (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Eta1stJet_gteTwoBtaggedJets"))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Jet #eta [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Eta2ndJet_gteTwoBtaggedJets"))
    plots[-1].rebin = 2
    # plots[-1].ymax = 200000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = -3
    plots[-1].xmax = 3
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet #eta [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Phi1stJet_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "1st Jet #phi [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Phi2ndJet_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e7
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "2nd Jet #phi [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("sTlep_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 3000
    plots[-1].xmin = 200
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (1st Electron, 2nd Electron) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("sTjet_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 200
    plots[-1].xmax = 3000
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (1st Jet, 2nd Jet) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("sT_gteTwoBtaggedJets", dataBlindAbove=dataBlindAboveSt))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 200.0
    plots[-1].xmax = 3000.0
    # plots[-1].rebin = 20
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "S_{T} (GeV) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("sT_zjj_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 200.
    # plots[-1].xmax = 2000.
    # plots[-1].rebin = 2
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "S_{T} (Z, 2jet) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet1_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jet 1 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet2_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jet 2 (GeV) [Preselection]"
    #
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Ele1_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from ele 1 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlo("sTfrac_Ele2_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from ele 2 (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Ele_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from electrons (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("sTfrac_Jet_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].xmin = 0.0
    # plots[-1].xmax = 1.0
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Fraction S_{T} from jets (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Mjj_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Dijet Mass (GeV) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("M_j2j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(j2,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_j1j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(j1,j3) Mass (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Me1j1_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 5e4
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e1,j1) Mass (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Me1j1_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 9.5e3
    # plots[-1].ymin = 1e-1
    plots[-1].rebin = 4
    plots[-1].xtit = "M(e1,j1) Mass (GeV) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("M_e1j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e1,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_e2j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e2,j3) Mass (GeV) [Preselection]"
    #
    # plots.append(makeDefaultPlot("M_eejjj_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].ymax = 20000
    # plots[-1].ymin = 1e-1
    # plots[-1].rebin = 4
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(e1, e2, j1, j2, j3) (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Meejj_gteTwoBtaggedJets"))
    plots[-1].rebin = 5
    # plots[-1].ymax = 6e5
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Mee_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    plots[-1].rebin = "var"
    plots[-1].xbins = list(range(0, 410, 10)) + [
        420,
        440,
        460,
        480,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
    ]
    # plots[-1].ymax = 2e4
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0.0
    plots[-1].xmax = 2000.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Mee_gteTwoBtaggedJets"))
    plots[-1].rebin = 4
    #plots[-1].rebin = "var"
    #plots[-1].xbins = range(0, 410, 10) + [
    #    420,
    #    440,
    #    460,
    #    480,
    #    500,
    #    550,
    #    600,
    #    650,
    #    700,
    #    800,
    #    900,
    #    1000,
    #    1200,
    #    1400,
    #    1600,
    #    1800,
    #    2000,
    #]
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 100.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, 100-250 GeV, >= 2 b-tags]"
    plots[-1].name = "Mee_100_250_gteTwoBtaggedJets"

    plots.append(makeDefaultPlot("Mee_gteTwoBtaggedJets"))
    plots[-1].ymax = 5e5
    plots[-1].ymin = 1e-1
    plots[-1].xmin = 60.0
    plots[-1].xmax = 250.0
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ee) (GeV) [Preselection, 60-250 GeV, >= 2 b-tags]"
    plots[-1].name = "Mee_60_250_gteTwoBtaggedJets"

    plots.append(makeDefaultPlot("Mee_EBEB_PAS_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EB-EB, >= 2 b-tags) (GeV)"

    plots.append(makeDefaultPlot("Mee_EBEE_PAS_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2e3
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EB-EE, >= 2 b-tags) (GeV)"

    plots.append(makeDefaultPlot("Mee_EEEE_PAS_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    #plots[-1].ymax = 1e3
    #plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xmin = 60.0
    plots[-1].xmax = 120.0
    plots[-1].xtit = "M(ee) (preselection, EE-EE, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EB_PAS_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 8e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) (preselection, EB-EE and EB-EB, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EBEB_80_100_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 8e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EB, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EBEE_80_100_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1500
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EE, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EEEE_80_100_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EE-EE, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EB_80_100_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 6e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [80, 100] (preselection, EB-EE and EB-EB, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_80_100_Preselection"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e8
    ## plots[-1].ymin = 1e-1
    #plots[-1].rebin = 4
    #plots[-1].xmin = 70.0
    #plots[-1].xmax = 110.0
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "M(ee) [80, 100] (GeV) [Preselection, >= 2 b-tags]"

    #plots.append(makeDefaultPlot("Mee_EBEB_70_110_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 1e4
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EB, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EBEE_70_110_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EE, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EEEE_70_110_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EE-EE, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_EB_70_110_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 6e3
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0.0
    #plots[-1].xmax = 1000.0
    #plots[-1].xtit = "M(ee) [70, 110] (preselection, EB-EE and EB-EB, >= 2 b-tags) (GeV)"

    #plots.append(makeDefaultPlot("Mee_70_110_Preselection"))
    #plots[-1].rebin = 1
    ## plots[-1].ymax = 2e8
    ## plots[-1].ymin = 1e-1
    #plots[-1].rebin = 4
    #plots[-1].xmin = 70.0
    #plots[-1].xmax = 110.0
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "M(ee) [70, 110] (GeV) [Preselection, >= 2 b-tags]"

    #plots.append(makeDefaultPlot("Ptee_gteTwoBtaggedJets"))
    #plots[-1].rebin = 2
    ## plots[-1].ymax = 4e6
    ## plots[-1].ymin = 1e-1
    #plots[-1].xmin = 0
    #plots[-1].xmax = 1000
    #plots[-1].ylog = "yes"
    #plots[-1].xtit = "P_{T}(ee) (GeV) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("Ptj1j2_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(j_{1}, j_{2}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj2j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{2}, j_{3}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj1j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{1}, j_{3}) (GeV) [Preselection]"
    #
    #
    # plots.append(makeDefaultPlot("Ptj1j2j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 1000
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(jet_{1}, jet_{2}, j_{3}) (GeV) [Preselection]"
    #
    #
    #
    # plots.append(makeDefaultPlot("Ptee_Minus_Ptj1j2_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -500
    # plots[-1].xmax  = 500
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(ee) - P_{T}(j_{1}, j_{2}) (GeV) [Preselection]"

    # plots.append(makeDefaultPlot("Ptee_Minus_Ptj1j2j3_gteTwoBtaggedJets"))
    # plots[-1].rebin = 2
    # plots[-1].ymax  = 2000000
    # plots[-1].ymin  = 1e-1
    # plots[-1].xmin  = -500
    # plots[-1].xmax  = 500
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit  = "P_{T}(ee) - P_{T}(j_{1}, j_{2}, j_{3}) (GeV) [Preselection]"

    plots.append(makeDefaultPlot("Me1j1_gteTwoBtaggedJets"))
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{1}j_{1}) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Me1j2_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{1}j_{2}) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Me2j1_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{2}j_{1}) (GeV) [Preselection, >= 2 b-tags]"

    plots.append(makeDefaultPlot("Me2j2_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(e_{2}j_{2}) (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    plots[-1].xmax = 2000
    plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_selected_avg_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1.8e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xtit = "M(ej) average (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1000
    plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_selected_min_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1.8e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xtit = "M(ej) minimum (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1000
    plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_selected_max_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1.8e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "no"
    plots[-1].xtit = "M(ej) maximum (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1000
    plots[-1].rebin = 5

    # USED IN AN
    plots.append(makeDefaultPlot("Mej_selected_avg_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 4e4
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ej) average (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1200
    plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_selected_min_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ej) minimum (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1200
    plots[-1].rebin = 5

    plots.append(makeDefaultPlot("Mej_selected_max_gteTwoBtaggedJets"))
    plots[-1].rebin = 1
    # plots[-1].ymax = 1000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "M(ej) maximum (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].xmin = 0
    # plots[-1].xmax = 1200
    plots[-1].rebin = 5

    # plots.append(makeDefaultPlot("Mej_selected_diff_gteTwoBtaggedJets")
    # plots[-1].rebin = 1
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "M(ej) diff (GeV) [Preselection]"
    # plots[-1].xmin  = 0
    # plots[-1].xmax  = 500
    # plots[-1].rebin = 5

    #plots.append(makeDefaultPlot("nVertex_gteTwoBtaggedJets"))
    #plots[-1].rebin = 1
    #plots[-1].xmin = -0.5
    #plots[-1].xmax = 60.5
    #plots[-1].ymin = 1e-1
    ## plots[-1].ymax = 5e7
    ## plots[-1].ylog = "yes"
    #plots[-1].xtit = "n(vertices) [Preselection, >= 2 b-tags]"

    # plots.append(makeDefaultPlot("nVertex_gteTwoBtaggedJets"))
    # plots[-1].rebin = 1
    # plots[-1].xmin = -0.5
    # plots[-1].xmax = 40.5
    # plots[-1].ymin = 1e-1
    # plots[-1].ymax = 6e2
    # plots[-1].xtit = "n(vertices) [Preselection]"

    # XXX REMOVE FOR NO JET REQUIREMENTS
    ##plots.append(makeDefaultPlot("DR_Ele1Jet1_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 1
    ##plots[-1].xtit = "#DeltaR(e_{1},j_{1})"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    ##plots[-1].ylog  = "yes"
    ##
    ##plots.append(makeDefaultPlot("DR_Ele1Jet2_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 1
    ##plots[-1].xtit = "#DeltaR(e_{1},j_{2})"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    ##plots[-1].ylog  = "yes"
    ##
    ##plots.append(makeDefaultPlot("DR_Ele2Jet1_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 1
    ##plots[-1].xtit = "#DeltaR(e_{2},j_{1})"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    ##plots[-1].ylog  = "yes"
    ##
    ##plots.append(makeDefaultPlot("DR_Ele2Jet2_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 1
    ##plots[-1].xtit = "#DeltaR(e_{2},j_{2})"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    ##plots[-1].ylog  = "yes"
    ##
    ##plots.append(makeDefaultPlot("DR_Ele1Ele2_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 4
    ##plots[-1].xtit = "#DeltaR(e_{1},e_{2})"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    ##plots[-1].ylog  = "yes"
    ##
    ##plots.append(makeDefaultPlot("minDR_EleJet_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    ##plots[-1].rebin = 1
    ##plots[-1].xtit = "Minimum #DeltaR(e_{1},(j_{1}, j_{2}, j_{3}))"
    ##plots[-1].ymax = 2000000
    ##plots[-1].ymin = 1e-1
    ##plots[-1].ylog  = "yes"
    ##plots[-1].xmin = 0
    ##plots[-1].xmax = 6
    #plots.append(makeDefaultPlot("minDR_ZJet_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 4
    # plots[-1].xtit = "Minimum #DeltaR(ee, (j_{1}, j_{2})) [Preselection]"
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xmin = 0
    # plots[-1].xmax = 6
    #
    #plots.append(makeDefaultPlot("DR_ZJet1_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 4
    # plots[-1].xtit = "#DeltaR(ee, j_{1}) [Preselection]"
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xmin = 0
    # plots[-1].xmax = 6
    #
    #plots.append(makeDefaultPlot("DR_ZJet2_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 4
    # plots[-1].xtit = "#DeltaR(ee, j_{2}) [Preselection]"
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xmin = 0
    # plots[-1].xmax = 6

    plots.append(makeDefaultPlot("Meejj_gteTwoBtaggedJets"))
    plots[-1].rebin = 10
    # plots[-1].ymax = 2000000
    # plots[-1].ymin = 1e-1
    plots[-1].ylog = "yes"
    plots[-1].xtit = "Mass_{eejj} (GeV) [Preselection, >= 2 b-tags]"

    #plots.append(makeDefaultPlot("Meej_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 10
    # plots[-1].ymax = 200000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Mass_{eej} (GeV) [Preselection]"

    #plots.append(makeDefaultPlot("Mejj_gteTwoBtaggedJets",histoBaseName_userDef,samplesForHistos,keys,samplesForStackHistos,keysStack,sampleForDataHisto,zUncBand,makeRatio))
    # plots[-1].rebin = 10
    # plots[-1].ymax = 200000
    # plots[-1].ymin = 1e-1
    # plots[-1].ylog  = "yes"
    # plots[-1].xtit = "Mass_{ejj} (GeV) [Preselection]"

    plots.append(makeDefaultPlot("MET_gteTwoBtaggedJets"))
    plots[-1].xtit = "PFMET (GeV) [Preselection, >= 2 b-tags]"
    plots[-1].rebin = 4
    # plots[-1].ymax = 1e6
    # plots[-1].ymin = 1e-1
    plots[-1].xmax = 500
    plots[-1].xmin = 0
    plots[-1].ylog = "yes"

    #plots.append(makeDefaultPlot2D_NoData("MeeVsST_PAS",histoBaseName2D_userDef,samplesForStackHistos,sampleForDataHisto))
    # plots[-1].xrebin = 2
    # plots[-1].yrebin = 2
    # plots[-1].xtit = "M(ee) [GeV]"
    # plots[-1].ytit = "S_{T}(eejj) [GeV]"
    # plots[-1].zlog = "yes"
    #
    #plots.append(makeDefaultPlot2D_NoData("MeeVsST_PAS",histoBaseName2D_userDef,samplesForStackHistos,sampleForDataHisto))
    # plots[-1].xrebin = 2
    # plots[-1].yrebin = 2
    # plots[-1].xtit = "M(ee) [GeV]"
    # plots[-1].ytit = "S_{T}(eejj) [GeV]"
    #
    #
    print("Done")

####################################################################################################
#  FINAL SELECTION
####################################################################################################
if doFinalSelectionPlots:
    print("INFO: creating final selection plots...", end=' ')
    for mass_point in LQmassesFinalSelection:
        samplesForHistos = [signalSampleName.format(mass_point)]
        keys = [signalSampleLabel.format(mass_point)]
        if blindFinalSelectionData:
            sampleForDataHisto = ""

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "nJet_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = (
                "Jets [LQ M = " + str(mass_point) + " selection]"
            )
            plots[-1].xtitPaper = "Jets"
            # plots[-1].rebin = 2
            plots[-1].ylog = "yes"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "nElectron_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = (
                "Electrons [LQ M = " + str(mass_point) + " selection]"
            )
            plots[-1].xtitPaper = "Electrons"
            # plots[-1].rebin = 2
            plots[-1].ylog = "yes"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Pt1stEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = (
            "1st Electron p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        )
        plots[-1].xtitPaper = "1st Electron p_{T} (GeV)"
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Pt2ndEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = (
            "2nd Electron p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        )
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Electron p_{T} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        # plots.append(
        #     makeDefaultPlot(
        #         "Pt1stMuon_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Muon p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        # )
        # plots[-1].rebin = 2

        # plots.append(
        #     makeDefaultPlot(
        #         "Pt1stMuon_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Muon p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        # )
        # plots[-1].rebin = 2

        plots.append(
            makeDefaultPlot(
                "Eta1stEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "1st Electron #eta [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "1st Electron #eta"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Eta2ndEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "2nd Electron #eta [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Electron #eta"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Phi1stEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "1st Electron #phi [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 4
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "1st Electron #phi"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Phi2ndEle_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "2nd Electron #phi [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 4
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Electron #phi"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Pt1stJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = (
            "1st Jet p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        )
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "1st Jet p_{T} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Pt2ndJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = (
            "2nd Jet p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
        )
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Jet p_{T} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Pt3rdJet_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = (
                "3rd Jet p_{T} (GeV) [LQ M = " + str(mass_point) + " selection]"
            )
            plots[-1].rebin = 2
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "3rd Jet p_{T} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Eta1stJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "1st Jet #eta [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "1st Jet #eta"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Eta2ndJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "2nd Jet #eta [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 2
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Jet #eta"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Eta3rdJet_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = "3rd Jet #eta [LQ M = " + str(mass_point) + " selection]"
            plots[-1].rebin = 2
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "3rd Jet #eta"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Phi1stJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "1st Jet #phi [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 4
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "1st Jet #phi"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "Phi2ndJet_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "2nd Jet #phi [LQ M = " + str(mass_point) + " selection]"
        plots[-1].rebin = 4
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "2nd Jet #phi"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Phi3rdJet_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = "3rd Jet #phi [LQ M = " + str(mass_point) + " selection]"
            plots[-1].rebin = 4
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "3rd Jet #phi"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        mej_rebin = 2
        mej_xmin = 0
        mej_xmax = 3000
        mee_rebin = 1
        st_rebin = 2
        dr_rebin = 2
        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Me1j1_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin * 5
            plots[-1].xtit = "M_{e1j1} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{e1j1} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Me1j2_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin * 5
            plots[-1].xtit = "M_{e1j2} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{e1j2} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Me2j1_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin * 5
            plots[-1].xtit = "M_{e2j1} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{e2j1} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Me2j2_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin * 5
            plots[-1].xtit = "M_{e2j2} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{e2j2} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Mej_selected_avg_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin
            plots[-1].xtit = (
                "M_{ej}^{avg} (GeV), (LQ M = " + str(mass_point) + " selection)"
            )
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{ej}^{avg} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            # plots.append ( makeDefaultPlot ( "Mej_selected_avg_LQ" + str ( mass_point) , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
            # plots[-1].rebin = mej_rebin
            # plots[-1].xtit = "M_{ej}^{avg} (GeV), (LQ M = " + str ( mass_point ) + " selection)"
            # plots[-1].xmin = mej_xmin
            # plots[-1].xmax = mej_xmax

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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].rebin = 4
        plots[-1].xtit = (
            "M_{ej}^{min} (GeV), (LQ M = " + str(mass_point) + " selection)"
        )
        plots[-1].xmin = mej_xmin
        plots[-1].xmax = mej_xmax
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "M_{ej}^{min} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymin = 1e-2
        plots[-1].ymax = 1e3
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Mej_selected_max_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin
            plots[-1].xtit = (
                "M_{ej}^{max} (GeV), (LQ M = " + str(mass_point) + " selection)"
            )
            plots[-1].xmin = mej_xmin
            plots[-1].xmax = mej_xmax
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{ej}^{max} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Mej_minmax_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin
            plots[-1].xtit = "M_{ej} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].ytit = "2 #times Entries"
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{ej} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Mej_asym_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = mej_rebin
            plots[-1].xtit = "M_{ej}^{asym} (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].ytit = "2 #times Entries"
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{ej}^{asym} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1


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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = (
            "M_{eejj} (GeV), (LQ M = " + str(mass_point) + " selection)"
        )
        # plots[-1].xmin = mej_xmin
        # plots[-1].xmax = mej_xmax
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "M_{eejj} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymin = 1e-2
        plots[-1].ymax = 1e5
        plots[-1].rebin = 20
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04

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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "S_{T} (GeV), (LQ M = " + str(mass_point) + " selection)"
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "S_{T} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
        plots[-1].ymin = 1e-2
        plots[-1].ymax = 1e5
        plots[-1].rebin = st_rebin
        if blindFinalSelectionData:
            plots[-1].labelSize = 0.04

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "sTlep_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = (
                "S_{T} (1st Electron, 2nd Electron) (GeV), (LQ M = "
                + str(mass_point)
                + " selection)"
            )
            plots[-1].rebin = 10
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "S_{T} (1st Electron, 2nd Electron) (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "sTjet_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].xtit = (
                "S_{T} (1st Jet, 2nd Jet) (GeV), (LQ M = " + str(mass_point) + " selection)"
            )
            plots[-1].rebin = 10
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "S_{T} (1st Jet, 2nd Jet) (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Jet1_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from jet 1 (GeV), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Jet2_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from jet 2 (GeV), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Ele1_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from ele 1 (GeV), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Ele2_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from ele 2 (GeV), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Ele_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from electrons (GeV), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "sTfrac_Jet_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = 0.0
        # plots[-1].xmax = 1.0
        # plots[-1].rebin = 4
        # plots[-1].xtit = (
        #     "Fraction S_{T} from jets (GeV), (LQ M = " + str(mass_point) + " selection)"
        # )

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "Ptee_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = 2
            plots[-1].xmin = 0
            plots[-1].xmax = 1000
            plots[-1].xtit = "P_{T}(ee) (GeV), (LQ M = " + str(mass_point) + " selection)"
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "P_{T}(ee) (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

            plots.append(
                makeDefaultPlot(
                    "Mjj_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = 4
            plots[-1].xtit = "Dijet Mass (GeV) (LQ M = " + str(mass_point) + " selection)"
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "M_{jj} (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        # plots.append(
        #     makeDefaultPlot(
        #         "Ptj1j2_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].rebin = 2
        # plots[-1].xmin = 0
        # plots[-1].xmax = 1000
        # plots[-1].xtit = (
        #     "P_{T}(j_{1}, j_{2}) (GeV), (LQ M = " + str(mass_point) + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "nVertex_LQ" + str(mass_point),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xmin = -0.5
        # plots[-1].xmax = 40.5
        # plots[-1].xtit = "n(vertexes) [Preselection]"

        plots.append(
            makeDefaultPlot(
                "Mee_LQ" + str(mass_point),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].rebin = mee_rebin
        plots[-1].xtit = "M(ee) (GeV), (LQ M = " + str(mass_point) + " selection)"
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "M_{ee} (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        if doFullSet:
            plots.append(
                makeDefaultPlot(
                    "DR_Ele1Jet1_LQ" + str(mass_point),
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
                    doPrefit=options.preFit,
                    fitType=options.fitType
                )
            )
            plots[-1].rebin = dr_rebin
            plots[-1].xtit = (
                "#DeltaR(e1,j1) (GeV), (LQ M = " + str(mass_point) + " selection)"
            )
            plots[-1].ylog = "yes"
            plots[-1].xtitPaper = "#DeltaR(e1,j1) (GeV)"
            plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
            plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
            plots[-1].makeNSigma = False if blindFinalSelectionData else True
            plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        # plots.append(
        #     makeDefaultPlot2D(
        #         "Mej_selected_min_vs_max_LQ" + str(mass_point),
        #         histoBaseName2D_userDef,
        #         samplesForStackHistos,
        #         sampleForDataHisto,
        #     )
        # )
        # plots[-1].xrebin = 2
        # plots[-1].yrebin = 2
        # plots[-1].xtit = "M(ej)^{min} (GeV), (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].ytit = "M(ej)^{max} (GeV), (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].ylog = "yes"
        # plots[-1].xtitPaper = "#DeltaR(e1,j1) (GeV)"
        # plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        # plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        # plots[-1].makeNSigma = False if blindFinalSelectionData else True
        # plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        # FIXME
        # plots.append(
        #     makeDefaultPlot2D_Ratio(
        #         "Mej_selected_min_vs_max_LQ" + str(mass_point),
        #         histoBaseName2D_userDef,
        #         samplesForStackHistos,
        #         sampleForDataHisto,
        #     )
        # )
        # plots[-1].xrebin = 2
        # plots[-1].yrebin = 2
        # plots[-1].xtit = "M(ej)^{min} (GeV), (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].ytit = "M(ej)^{max} (GeV), (LQ M = " + str(mass_point) + " selection)"

        # FIXME
        # plots.append(
        #     makeDefaultPlot2D_NSigma(
        #         "Mej_selected_min_vs_max_LQ" + str(mass_point),
        #         histoBaseName2D_userDef,
        #         samplesForStackHistos,
        #         sampleForDataHisto,
        #     )
        # )
        # plots[-1].xrebin = 2
        # plots[-1].yrebin = 2
        # plots[-1].xtit = "M(ej)^{min} (GeV), (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].ytit = "M(ej)^{max} (GeV), (LQ M = " + str(mass_point) + " selection)"

        # plots.append ( makeDefaultPlot ("Classif_1stEle_LQ" + str ( int ( mass_point )  )               , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron classification, (LQ M = " + str ( mass_point ) + " selection)"
        #
        #
        # plots.append ( makeDefaultPlot ("Classif_2ndEle_LQ" + str ( int ( mass_point )  )               , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron classification, (LQ M = " + str ( mass_point ) + " selection)"

        # plots.append(
        #     makeDefaultPlot(
        #         "CorrIsolation_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron corrected HEEP isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmax = 1.0
        # plots[-1].xmin = -25
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "CorrIsolation_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron corrected HEEP isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmax = 1.0
        # plots[-1].xmin = -25
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "DeltaEtaTrkSC_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron #Delta#eta(track, SC), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = -0.007
        # plots[-1].xmax = 0.007
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "DeltaEtaTrkSC_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron #Delta#eta(track, SC), (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = -0.007
        # plots[-1].xmax = 0.007
        # plots[-1].rebin = 4

        # plots.append ( makeDefaultPlot ("DeltaPhiTrkSC_1stEle_LQ" + str ( int ( mass_point )  )         , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron #Delta#phi(track, SC), (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = -0.06
        # plots[-1].xmax  =  0.06
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("DeltaPhiTrkSC_2ndEle_LQ" + str ( int ( mass_point )  )         , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron #Delta#phi(track, SC), (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = -0.06
        # plots[-1].xmax  =  0.06
        # plots[-1].rebin = 4

        # plots.append ( makeDefaultPlot ("Full5x5E1x5OverE5x5_1stEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron E_{1x5} / E_{5x5}, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =  0.20
        # plots[-1].xmax  =  1.40
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("Full5x5E1x5OverE5x5_2ndEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron E_{1x5} / E_{5x5}, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =  0.20
        # plots[-1].xmax  =  1.40
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("Full5x5E2x5OverE5x5_1stEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron E_{2x5} / E_{5x5}, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =  0.20
        # plots[-1].xmax  =  1.40
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("Full5x5E2x5OverE5x5_2ndEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron E_{2x5} / E_{5x5}, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =  0.20
        # plots[-1].xmax  =  1.40
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "EcalIsolation_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron HEEP ECAL isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = 0.00
        # plots[-1].xmax = 15.00
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "EcalIsolation_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron HEEP ECAL isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = 0.00
        # plots[-1].xmax = 15.00
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "HcalIsolation_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron HEEP HCAL isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = 0.00
        # plots[-1].xmax = 15.00
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "HcalIsolation_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron HEEP HCAL isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = 0.00
        # plots[-1].xmax = 15.00
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "TrkIsolation_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron HEEP tracker isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "TrkIsolation_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron HEEP tracker isolation, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append ( makeDefaultPlot ("FBrem_1stEle_LQ" + str ( int ( mass_point )  )                 , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron brem fraction, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =   -5.0
        # plots[-1].xmax  =    4.0
        #
        # plots.append ( makeDefaultPlot ("FBrem_2ndEle_LQ" + str ( int ( mass_point )  )                 , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron brem fraction, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  =   -5.0
        # plots[-1].xmax  =    4.0
        #
        # plots.append ( makeDefaultPlot ("GsfCtfCharge_1stEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron GSF CTF charge, (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("GsfCtfCharge_2ndEle_LQ" + str ( int ( mass_point )  )          , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron GSF CTF charge, (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("GsfCtfScPixCharge_1stEle_LQ" + str ( int ( mass_point )  )     , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron GSF CTF SC Pixel charge, (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("GsfCtfScPixCharge_2ndEle_LQ" + str ( int ( mass_point )  )     , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron GSF CTF SC Pixel charge, (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("GsfScPixCharge_1stEle_LQ" + str ( int ( mass_point )  )        , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron GSF SC Pixel charge, (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("GsfScPixCharge_2ndEle_LQ" + str ( int ( mass_point )  )        , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron GSF SC Pixel charge, (LQ M = " + str ( mass_point ) + " selection)"

        # plots.append(
        #     makeDefaultPlot(
        #         "HasMatchedPhot_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron has matched photon, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "HasMatchedPhot_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron has matched photon, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "HoE_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = "1st Electron H/E, (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "HoE_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = "2nd Electron H/E, (LQ M = " + str(mass_point) + " selection)"
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "LeadVtxDistXY_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron D_{XY} vs leading vertex, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "LeadVtxDistXY_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron D_{XY} vs leading vertex, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "LeadVtxDistZ_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron D_{Z} vs leading vertex, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "LeadVtxDistZ_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron D_{Z} vs leading vertex, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "MissingHits_1stEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "1st Electron N(missing hits), (LQ M = " + str(mass_point) + " selection)"
        # )

        # plots.append(
        #     makeDefaultPlot(
        #         "MissingHits_2ndEle_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "2nd Electron N(missing hits), (LQ M = " + str(mass_point) + " selection)"
        # )

        # plots.append ( makeDefaultPlot ("NBrems_1stEle_LQ" + str ( int ( mass_point )  )                , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron N(brems), (LQ M = " + str ( mass_point ) + " selection)"
        #
        # plots.append ( makeDefaultPlot ("NBrems_2ndEle_LQ" + str ( int ( mass_point )  )                , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron N(brems), (LQ M = " + str ( mass_point ) + " selection)"

        # plots.append ( makeDefaultPlot ("EnergyORawEnergy_1stEle_LQ" + str ( int ( mass_point )  )      , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron energy correction factor, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = 0.95
        # plots[-1].xmax  = 1.3
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("EnergyORawEnergy_2ndEle_LQ" + str ( int ( mass_point )  )      , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron energy correction factor, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = 0.95
        # plots[-1].xmax  = 1.3
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("SigmaEtaEta_Barrel_1stEle_LQ" + str ( int ( mass_point )  )    , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron #sigma_{#eta#eta} [Preselection, Barrel]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.015
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("SigmaEtaEta_Barrel_2ndEle_LQ" + str ( int ( mass_point )  )    , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron #sigma_{#eta#eta} [Preselection, Barrel]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.015
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("SigmaEtaEta_Endcap_1stEle_LQ" + str ( int ( mass_point )  )    , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron #sigma_{#eta#eta} [Preselection, Endcap]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.04
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("SigmaEtaEta_Endcap_2ndEle_LQ" + str ( int ( mass_point )  )    , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron #sigma_{#eta#eta} [Preselection, Endcap]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.04
        # plots[-1].rebin = 2

        # plots.append ( makeDefaultPlot ("Full5x5SigmaIEtaIEta_Barrel_1stEle_LQ" + str ( int ( mass_point )  )  , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron #sigma_{i#etai#eta} [Preselection, Barrel]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.015
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("Full5x5SigmaIEtaIEta_Barrel_2ndEle_LQ" + str ( int ( mass_point )  )  , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron #sigma_{i#etai#eta} [Preselection, Barrel]"
        # plots[-1].xmin  = 0.005
        # plots[-1].xmax  = 0.015
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("Full5x5SigmaIEtaIEta_Endcap_1stEle_LQ" + str ( int ( mass_point )  )  , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron #sigma_{i#etai#eta} [Preselection, Endcap]"
        # plots[-1].xmin  = 0.015
        # plots[-1].xmax  = 0.035
        # plots[-1].rebin = 2
        #
        # plots.append ( makeDefaultPlot ("Full5x5SigmaIEtaIEta_Endcap_2ndEle_LQ" + str ( int ( mass_point )  )  , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron #sigma_{i#etai#eta} [Preselection, Endcap]"
        # plots[-1].xmin  = 0.015
        # plots[-1].xmax  = 0.035
        # plots[-1].rebin = 2

        # plots.append ( makeDefaultPlot ("TrkPtOPt_1stEle_LQ" + str ( int ( mass_point )  )              , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron track pt / SC pt, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("TrkPtOPt_2ndEle_LQ" + str ( int ( mass_point )  )              , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron track pt / SC pt, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("ValidFrac_1stEle_LQ" + str ( int ( mass_point )  )             , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "1st Electron valid fraction of hits, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = 0.00
        # plots[-1].xmax  = 1.50
        # plots[-1].rebin = 4
        #
        # plots.append ( makeDefaultPlot ("ValidFrac_2ndEle_LQ" + str ( int ( mass_point )  )              , histoBaseName_userDef, samplesForHistos, keys, samplesForStackHistos, keysStack, sampleForDataHisto, zUncBand, makeRatio) )
        # plots[-1].xtit  = "2nd Electron valid fraction of hits, (LQ M = " + str ( mass_point ) + " selection)"
        # plots[-1].xmin  = 0.00
        # plots[-1].xmax  = 1.50
        # plots[-1].rebin = 4

        # plots.append(
        #     makeDefaultPlot(
        #         "EleChargeSum_LQ" + str(int(mass_point)),
        #         histoBaseName_userDef,
        #         samplesForHistos,
        #         keys,
        #         samplesForStackHistos,
        #         keysStack,
        #         sampleForDataHisto,
        #         zUncBand,
        #         makeRatio,
        #     )
        # )
        # plots[-1].xtit = (
        #     "Electron 1 charge + Electron 2 charge, (LQ M = "
        #     + str(mass_point)
        #     + " selection)"
        # )
        # plots[-1].xmin = -1.5
        # plots[-1].xmax = 1.5

        plots.append(
            makeDefaultPlot(
                "MET_LQ" + str(int(mass_point)),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "PFMET (GeV), (LQ M = " + str(mass_point) + " selection)"
        plots[-1].rebin = 4
        plots[-1].xmax = 500
        plots[-1].xmin = 0
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "PFMET (GeV)"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1

        plots.append(
            makeDefaultPlot(
                "BDTOutput_LQ" + str(int(mass_point)),
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
                doPrefit=options.preFit,
                fitType=options.fitType
            )
        )
        plots[-1].xtit = "BDT score, (LQ M = " + str(mass_point) + " selection)"
        plots[-1].rebin = 50
        #plots[-1].xmax = 500
        #plots[-1].xmin = 0
        plots[-1].ylog = "yes"
        plots[-1].xtitPaper = "BDT score"
        plots[-1].extraText = "m_{{LQ}} = {} GeV".format(mass_point)
        plots[-1].makeRatio = False if blindFinalSelectionData else makeRatio
        plots[-1].makeNSigma = False if blindFinalSelectionData else True
        plots[-1].dataBlindAbove = 0 if blindFinalSelectionData else -1
    print("Done")

####################################################################################################
# --- Generate and print the plots from the list 'plots' define above
####################################################################################################
CMS.setCMSStyle()
c = TCanvas()

plotsAN = copy.deepcopy(plots)
plotsPaper = copy.deepcopy(plots)
fileps = "allPlots_eejj_analysis.ps"
print("INFO: writing canvas with plots to PDF...", end=' ')
c.Print(fileps + "[")
for i_plot, plot in enumerate(plotsAN):
    print()
    print("INFO: draw plot:", plot.name, flush=True)
    print()
    try: 
        plot.Draw(fileps, i_plot + 1)
    except Exception as e:
        raise RuntimeError("Caught exception while making plot", plot.name, ":", e)
c.Print(fileps + "]")
print("DONE")
print("INFO: MakeTOC()")
makeTOC("allPlots_eejj_analysis_toc.tex", fileps, plotsAN)

c = TCanvas()
fileps = "allPlots_eejj_paper.ps"
print("INFO: Doing paper plots")
print("INFO: writing canvas with plots to PDF...", end=' ')
c.Print(fileps + "[")
for i_plot, plot in enumerate(plotsPaper):
    try:
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
