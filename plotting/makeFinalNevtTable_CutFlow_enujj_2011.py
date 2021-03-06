#!/usr/bin/env python

import pprint # for pretty printing
import math

## 2011 pre-approval ##
# data files (V00-01-04 to V00-01-06 MC ntuples, enujj skim, jetid flags + QCD fake rate for Njet>=2 and full dataset)
f1 = open("/home/ferencek/work/Leptoquarks/output_fromAFS/enujj_analysis/36.0pb-1_presel_MET45_presel_sT250_Wrescale1.18_Feb112011/analysisClass_enujjSample_tables.dat")
f2 = open("/home/ferencek/work/Leptoquarks/output_fromAFS/enujj_analysis/35.8pb-1_QCD_UseHLTPrescales_presel_MET45_presel_sT250_Feb112011/analysisClass_enujjSample_QCD_tables.dat")
# data files (V00-01-04 to V00-01-06 MC ntuples, enujj skim, jetid flags)
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/36.0pb-1_sT_presel_250_Zrescale1.20_Wrescale1.19_enujjskim_MET45_Jan11Prod/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/35.8pb-1_QCD_sT_presel_250_UseHLTPrescales_enujjskim_MET45/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")
# data files (V00-01-04 to V00-01-06 MC ntuples)
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/36.0pb-1_sT_presel_250_Zrescale1.20_Wrescale1.19_fullntuples_MET45_Jan11Prod/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/35.8pb-1_QCD_sT_presel_250_UseHLTPrescales_fullntuples_MET45/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")
# data files (V00-01-04 to V00-01-06 MC ntuples + type1 PFMET)
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/36.0pb-1_sT_presel_250_Zrescale1.20_Wrescale1.19_fullntuples_MET45_Jan11Prod_type1PFMET/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/35.8pb-1_QCD_sT_presel_250_UseHLTPrescales_fullntuples_MET45_type1PFMET/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")
# data files (V00-00-XX MC ntuples)
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/36.0pb-1_sT_presel_250_Zrescale1.20_Wrescale1.19_fullntuples_MET45/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/35.8pb-1_QCD_sT_presel_250_UseHLTPrescales_fullntuples_MET45/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")


## Dec 2010 pre-approval ##
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/33.2pb-1_sT_presel_250_Zrescale1.20_Wrescale1.06_extraPlotsDec9/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/6.1pb-1_QCD_HLT30_sT_presel_250_extraPlotsDec9/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")
#f1 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/33.2pb-1_sT_presel_250_Zrescale1.20_Wrescale1.06/output_cutTable_enujjSample/analysisClass_enujjSample_tables.dat")
#f2 = open("/home/santanas/Leptoquarks/data/output_fromAFS/enujj_analysis/6.1pb-1_QCD_HLT30_sT_presel_250/output_cutTable_enujjSample_QCD/analysisClass_enujjSample_QCD_tables.dat")

#dict containing all values
d = {}

###########
# User data
###########

## List of cuts
cutNames = [ "Pt1stEle_PAS",
             "MTenu",
             "minMETPt1stEle",
             "sT_MLQ300",
           ]
cutLabels = [ r"\enujj pre-sel.",
              r"\mt$>125$ \GeVmass",
              r"\minptmet$>85$ \GeVmom",
              r"\st$>490$ \GeVmom",
            ]

LQmassInTable = "LQenujj_M300"

## List of samples
blocks = { 'all' : {"ALLBKG":         {"rescale": 0.001, "label":  "All Bkgs"},
                    "TTbar_Madgraph": {"rescale": 0.001, "label": r"$t\bar{t}$ + jets"},
                    "WJetAlpgen":     {"rescale": 0.001, "label": r"$W$ + jets"},
                    "OTHERBKG":       {"rescale": 0.001, "label":  "Other Bkgs"},
                    "DATA":           {"rescale": 1,     "label":  "DATA"},
                    "LQenujj_M200":   {"rescale": 0.001, "label":  "LQenujj200"},
                    "LQenujj_M250":   {"rescale": 0.001, "label":  "LQenujj250"},
                    "LQenujj_M280":   {"rescale": 0.001, "label":  "LQenujj280"},
                    "LQenujj_M300":   {"rescale": 0.001, "label":  "LQenujj300"},
                    "LQenujj_M320":   {"rescale": 0.001, "label":  "LQenujj320"},
                    "LQenujj_M340":   {"rescale": 0.001, "label":  "LQenujj340"},
                    "LQenujj_M370":   {"rescale": 0.001, "label":  "LQenujj370"},
                    "LQenujj_M400":   {"rescale": 0.001, "label":  "LQenujj400"},
                    "LQenujj_M450":   {"rescale": 0.001, "label":  "LQenujj450"},
                    "LQenujj_M500":   {"rescale": 0.001, "label":  "LQenujj500"},
                    },
           'QCD' : {"DATA":           {"rescale": 1,     "label":  "QCD"},
                    }
#           'QCD' : {"DATA":           {"rescale": 5.4, "label": "QCD"},
#                    }
         }


################################################
### HOW TO SORT A DICT ACCORDINGLY WITH THE ORIGINAL ORDERING ??
#cutNames = { "Pt1stEle_PAS": r"$e\nu jj$ pre-sel.",
#             "nMuon_PtCut_IDISO": r"\nmuon$=0$",
#             "minMETPt1stEle": r"\minptmet$>85$ \GeVmom",
#             "MTenu": r"\mt$>125$ \GeVmass",
#             "sT_MLQ280": r"\st$>460$ \GeVmom",
#           }
#cutNames = { "Pt1stEle_PAS":        {"label": r"$e\nu jj$ pre-sel.", "level": 1},
#             "nMuon_PtCut_IDISO":   {"label": r"\nmuon$=0$"        , "level": 2},
#             "minMETPt1stEle":      {"label": r"\minptmet$>85$ \GeVmom", "level": 3},
#             "MTenu":               {"label": r"\mt$>125$ \GeVmass", "level": 4},
#             "sT_MLQ280":           {"label": r"\st$>460$ \GeVmom", "level": 5},
#           }
################################################



###########
# Algorithm
###########
for sample in blocks:
  if sample == "all": ff = f1
  if sample == "QCD": ff = f2
  yesBlock = False
  for ln, line in enumerate(ff): #.readlines()):
        #print "line no.", ln
        line = line.strip()
        if line in blocks[sample]:
            yesBlock = True
            block = line
            continue
        if yesBlock:
            if line == "":
              yesBlock = False
              continue
            col = line.split()
            cutName = col[0].strip()
            #print cutName
            if cutName in cutNames:

                cut = float(col[1].strip())
                Npass = float(col[5].strip())
                errNpass = float(col[6].strip())
                EffAbs = float(col[9].strip())
                errEffAbs = float(col[10].strip())

                if not cutName in d:
                    d[cutName] = {}
                if not sample in d[cutName]:
                    d[cutName][sample] = {}
                if not block in d[cutName][sample]:
                    d[cutName][sample][block] = {}
                d[cutName][sample][block]['cut'] = cut
                d[cutName][sample][block]['Npass'] = Npass * blocks[sample][block]['rescale']
                d[cutName][sample][block]['errNpass'] = errNpass * blocks[sample][block]['rescale']
                d[cutName][sample][block]['EffAbs'] = EffAbs
                d[cutName][sample][block]['errEffAbs'] = errEffAbs



######################
# Operations on values
######################

for cut in cutNames:
  if not 'ALLBKG+QCD' in d[cut]['all']:
    d[cut]['all']['ALLBKG+QCD'] = {}
  d[cut]['all']['ALLBKG+QCD']['Npass']=d[cut]['all']['ALLBKG']['Npass'] + d[cut]['QCD']['DATA']['Npass']
  d[cut]['all']['ALLBKG+QCD']['errNpass']=math.sqrt( d[cut]['all']['ALLBKG']['errNpass']**2
                                                     + (d[cut]['QCD']['DATA']['errNpass'])**2 )



######################################
# Print the dictionary with all values
######################################
pprint.pprint(d)


############################
# Output on LaTeX table
############################
fout = open("table_CutFlow_enujj.tex", "w")
#fout.write(r"\begin{table}[]"+"\n")
for idx, cutName in enumerate(cutNames):
  print cutName
  fout.write(r" %s & %.2f$\pm$%.2f & %.3f & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %i \\" %
             (
               #cutNames[cutName]['label'],
               cutLabels[idx],
               d[cutName]['all'][LQmassInTable]['Npass'],
               d[cutName]['all'][LQmassInTable]['errNpass'],
               d[cutName]['all'][LQmassInTable]['EffAbs'],
               #d[cutName]['all'][LQmassInTable]['errEffAbs'],
               d[cutName]['all']["TTbar_Madgraph"]['Npass'],
               d[cutName]['all']["TTbar_Madgraph"]['errNpass'],
               d[cutName]['all']["WJetAlpgen"]['Npass'],
               d[cutName]['all']["WJetAlpgen"]['errNpass'],
               d[cutName]['all']["OTHERBKG"]['Npass'],
               d[cutName]['all']["OTHERBKG"]['errNpass'],
               d[cutName]['QCD']["DATA"]['Npass'],
               d[cutName]['QCD']["DATA"]['errNpass'],
               d[cutName]['all']["ALLBKG+QCD"]['Npass'],
               d[cutName]['all']["ALLBKG+QCD"]['errNpass'],
               d[cutName]['all']["DATA"]['Npass'],
               ) + "\n"
             )

fout.close()




