#!/usr/bin/env python

##############################################################################
## USER CODE IS TOWARD THE END OF THE FILE
##############################################################################

##############################################################################
############# DON'T NEED TO MODIFY ANYTHING HERE - BEGIN #####################

#---Import
import sys
import string
from optparse import OptionParser
import os.path
from ROOT import *
import re
from array import array
import copy
import math

#--- ROOT general options
gROOT.SetBatch(kTRUE);
gStyle.SetOptStat(0)
gStyle.SetPalette(1)
gStyle.SetCanvasBorderMode(0)
gStyle.SetFrameBorderMode(0)
gStyle.SetCanvasColor(kWhite)
gStyle.SetPadTickX(1);
gStyle.SetPadTickY(1);
#--- TODO: WHY IT DOES NOT LOAD THE DEFAULT ROOTLOGON.C ? ---#

def GetFile(filename):
    file = TFile(filename)
    if( not file):
        print "ERROR: file " + filename + " not found"
        print "exiting..."
        sys.exit()
    return file


def GetHisto( histoName , file, scale = 1 ):
    histo = file.Get( histoName )
    if( not histo):
        print "ERROR: histo " + histoName + " not found in " + file.GetName()
        print "exiting..."
        sys.exit()
    new = copy.deepcopy(histo)
    if(scale!=1):
        new.Scale(scale)
    return new

def GetIntegralTH1( histo, xmin, xmax):
    #get integral
    axis = histo.GetXaxis()
    bmin = axis.FindBin(xmin)
    bmax = axis.FindBin(xmax)
    bminResidual = histo.GetBinContent(bmin)*(xmin-axis.GetBinLowEdge(bmin)) / axis.GetBinWidth(bmin)
    bmaxResidual = histo.GetBinContent(bmax)*(axis.GetBinUpEdge(bmax)-xmax) / axis.GetBinWidth(bmax)
    integral = histo.Integral(bmin,bmax) - bminResidual - bmaxResidual
    return integral

def GetErrorIntegralTH1( histo, xmin, xmax):
    print "## calculating error for integral of histo " + str(histo)
    print "## in the x range [" + str(xmin) + "," + str(xmax) + "]"
    #get integral
    axis = histo.GetXaxis()
    bmin = axis.FindBin(xmin)
    bmax = axis.FindBin(xmax)
    bminResidual = histo.GetBinContent(bmin)*(xmin-axis.GetBinLowEdge(bmin)) / axis.GetBinWidth(bmin)
    bmaxResidual = histo.GetBinContent(bmax)*(axis.GetBinUpEdge(bmax)-xmax) / axis.GetBinWidth(bmax)
    integral = histo.Integral(bmin,bmax) - bminResidual - bmaxResidual
    error = 0
    for bin in range(bmin, bmax+1):
	#print "bin: " +str(bin)
        if(bin==bmax and bmaxResidual==histo.GetBinContent(bmax)): # skip last bin if out of range
            print "     --> skip bin: " + str(bin)
        else:
            error = error + histo.GetBinError(bin)**2
            #print "error**2 : " + str(error)

    error = math.sqrt(error)
    print  " "
    return error


## The Plot class: add members if needed
class Plot:
    histoDATA    = "" # DATA
    histoMCTTbar = "" # MCTTbar
    histoMCall   = "" # MCall
    histoQCD     = "" # QCD
    histoZJet    = ""
    histoWJet    = ""
    histoSingleTop = ""
    histoPhotonJets = ""
    histoDiboson = ""
    xtit         = "" # xtitle
    ytit         = "" # ytitle
    xmin         = "" # set xmin to calculate rescaling factor (-- please take into account the bin size of histograms --)
    xmax         = "" # # set xmax to calculate rescaling factor (-- please take into account the bin size of histograms --)
    xminplot     = "" # min x axis range (need to set both min and max. Leave it as is for full range)
    xmaxplot     = "" # max x axis range (need to set both min and max. Leave it as is for full range)
    yminplot     = "" # min y axis range (need to set both min and max. Leave it as is for full range)
    ymaxplot     = "" # max y axis range (need to set both min and max. Leave it as is for full range)
    lpos         = "" # legend position (default = top-right, option="bottom-center", "top-left")
    #    xlog         = "" # log scale of X axis (default = no, option="yes") ### IT SEEMS IT DOES NOT WORK
    ylog         = "" # log scale of Y axis (default = no, option="yes")
    #rebin       = "" # rebin x axis (default = 1, option = set it to whatever you want )
    name         = "" # name of the final plots
    lint         = "2.6 fb^{-1}" # integrated luminosity of the sample ( example "10 pb^{-1}" )
    fileXsectionNoRescale = "" #cross section file (with no rescale
    datasetName = "" # string for pattern recognition of dataset name (rescaling will be done only on matched datasets)

    def CalculateRescaleFactor(self, fileps):
        #calculate rescaling factor for Z/gamma+jet background and create new cross section file
        canvas = TCanvas()

        #check
        if(self.histoMCall.GetNbinsX()!=self.histoDATA.GetNbinsX()):
            print "WARNING! number of bins is different between DATA and MC"
            print "exiting..."
            sys.exit()
        if(self.histoMCall.GetBinWidth(1)!=self.histoDATA.GetBinWidth(1)):
            print "WARNING! bin width is different between DATA and MC"
            print "exiting..."
            sys.exit()

        #integrals
        integralDATA = GetIntegralTH1(self.histoDATA,self.xmin,self.xmax)
        ERRintegralDATA = GetErrorIntegralTH1(self.histoDATA,self.xmin,self.xmax)
        integralMCall = GetIntegralTH1(self.histoMCall,self.xmin,self.xmax)
        ERRintegralMCall = GetErrorIntegralTH1(self.histoMCall,self.xmin,self.xmax)
        integralMCTTbar = GetIntegralTH1(self.histoMCTTbar,self.xmin,self.xmax)
        ERRintegralMCTTbar = GetErrorIntegralTH1(self.histoMCTTbar,self.xmin,self.xmax)
        integralQCD = GetIntegralTH1(self.histoQCD,self.xmin,self.xmax)
        ERRintegralQCD = GetErrorIntegralTH1(self.histoQCD,self.xmin,self.xmax)

        #contamination from other backgrounds (except TTbar) in the integral range
        integralMCothers = integralMCall - integralMCTTbar + integralQCD
        ERRintegralMCothers = math.sqrt(ERRintegralMCall**2 + ERRintegralMCTTbar**2 + ERRintegralQCD**2)
        contamination = integralMCothers / integralMCall

        #DATA corrected for other bkg contamination --> best estimate of DATA (due to Z only)
        integralDATAcorr = (integralDATA - integralMCothers)
        ERRintegralDATAcorr = math.sqrt(ERRintegralDATA**2 + ERRintegralMCothers**2)

        #rescale factor
        rescale = integralDATAcorr / integralMCTTbar
        relERRintegralDATAcorr = ERRintegralDATAcorr / integralDATAcorr
        relERRintegralMCTTbar = ERRintegralMCTTbar / integralMCTTbar
        relERRrescale = math.sqrt(relERRintegralDATAcorr**2 + relERRintegralMCTTbar**2)

        #draw histo
        self.histoMCall.SetFillColor(kBlue)
        self.histoDATA.SetMarkerStyle(20)

        self.histoMCall.Draw("HIST")
        self.histoDATA.Draw("psame")
        self.histoMCall.GetXaxis().SetRangeUser(self.xminplot,self.xmaxplot)
        self.histoMCall.GetYaxis().SetRangeUser(self.yminplot,self.ymaxplot)

        canvas.Update()
        gPad.RedrawAxis()
        gPad.Modified()
        #canvas.SaveAs(self.name + ".eps","eps")
        #canvas.SaveAs(self.name + ".pdf","pdf")
        canvas.Print(fileps)
        canvas.Print(self.name + ".C")
        # make root file
        tfile = TFile(self.name+'.root','recreate')
        tfile.cd()
        self.histoDATA.Write()
        self.histoMCTTbar.Write()
        self.histoMCall.Write()
        self.histoQCD.Write()
        self.histoZJet.Write()
        self.histoWJet.Write()
        self.histoSingleTop.Write()
        self.histoPhotonJets.Write()
        self.histoDiboson.Write()
        tfile.Close()

        #printout
        print " "
        print "######################################## "
        print "name:                " + self.name
        print "integral range:      " + str(self.xmin) + " < Mee < " + str(self.xmax) + " GeV/c2"
        print "integral MC All:     "   + str( integralMCall ) + " +/- " + str( ERRintegralMCall )
        print "integral QCD:        "   + str( integralQCD ) + " +/- " + str( ERRintegralQCD )
        print "integral MC TTbar:   "   + str( integralMCTTbar) + " +/- " + str( ERRintegralMCTTbar )
        print "integral DATA:       "   + str( integralDATA ) + " +/- " + str( ERRintegralDATA )
        print "contribution from other bkgs (except TTbar): " + str(contamination*100) + "%"
        print "integral DATA (corrected for contribution from other bkgs): "  + str( integralDATAcorr ) + " +/- " + str( ERRintegralDATAcorr )
        print "rescale factor for TTbar background: " + str(rescale) + " +\- " + str(relERRrescale*rescale)
        print "systematical uncertainty of TTbar background modeling: " + str(relERRrescale*100) + "%"
        print "######################################## "
        print " "

        #create new cross section file
        originalFileName = string.split( string.split(self.fileXsectionNoRescale, "/" )[-1], "." ) [0]
        newFileName = originalFileName + "_" + self.name +".txt"
        os.system('rm -f '+ newFileName)
        outputFile = open(newFileName,'w')

        for line in open( self.fileXsectionNoRescale ):
            line = string.strip(line,"\n")

            if( re.search(self.datasetName, line) ):
                list = re.split( '\s+' , line  )
                newline = str(list[0]) + "    "  + str("%.6f" % (float(list[1])*float(rescale)) )
                print >> outputFile, newline
            else:
                print >> outputFile, line

        outputFile.close
        print "New xsection file (after TTbar rescaling) is: " + newFileName
        print " "


############# DON'T NEED TO MODIFY ANYTHING HERE - END #######################
##############################################################################


##############################################################################
############# USER CODE - BEGING #############################################

#--- Input files
#File_preselection = GetFile("$LQDATA/RunII/eejj_analysis_ttbarRescaleFinalSels_2jun2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots_noTTbarRescale.root")
#File_QCD_preselection = GetFile("$LQDATA/RunII/eejj_analysis_ttbarRescaleFinalSels_2jun2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_QCD_plots_noTTbarRescale.root")
#File_preselection = GetFile("$LQDATA/RunII/eejj_analysis_MeePlotsWithStCuts_13jul2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots.root")
#File_QCD_preselection = GetFile("$LQDATA/RunII/eejj_analysis_MeePlotsWithStCuts_13jul2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_QCD_plots.root")
File_preselection = GetFile("$LQDATA/2016analysis/oct27_addStSFplots_allDataAndMC_ele27wptightOrPhoton175Data2015CurveMC_eejj2015FinalSels/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots.root")
File_QCD_preselection = GetFile("$LQDATA/2016analysis/oct27_addStSFplots_allDataAndMC_ele27wptightOrPhoton175Data2015CurveMC_eejj2015FinalSels/output_cutTable_lq_eejj/analysisClass_lq_eejj_QCD_plots.root")

#--- Rescaling of Z/gamma + jet background

#-----------------------------------------
#FIXME these aren't right, correct if going to use
#h_ALLBKG_Mee = GetHisto("histo1D__ALLBKG__cutHisto_allPreviousCuts________Mee", File_preselection) # MC all
# because M_e1e2 is the last cut applied, all passing all previous cuts is the same as passing all other cuts
#h_ALLBKG_Mee = GetHisto("histo1D__ALLBKG__Mee_80_100_Preselection", File_preselection) # MC all
#h_ZJetMadgraph_Mee = GetHisto("histo1D__ZJet_Madgraph_HT__Mee_80_100_Preselection", File_preselection) # MC Z
## amcatnlo
#h_ALLBKG_Mee = GetHisto("histo1D__ALLBKG_amcAtNLOInc_TTBar__Mee_80_100_Preselection", File_preselection) # MC all
#h_ZJetMadgraph_Mee = GetHisto("histo1D__ZJet_amcatnlo_Inc__Mee_80_100_Preselection", File_preselection) # MC Z
## MG Inc
#h_ALLBKG_Mee = GetHisto("histo1D__ALLBKG_MGInc__Mee_80_100_Preselection", File_preselection) # MC all
#h_ZJetMadgraph_Mee = GetHisto("histo1D__ZJet_Madgraph_Inc__Mee_80_100_Preselection", File_preselection) # MC Z

histBaseNames = []
# nominal
histBaseNames.append('Mee_PAS')
##histBaseNames.append('Mee_NJetEq2_PAS')
##histBaseNames.append('Mee_NJetEq3_PAS')
##histBaseNames.append('Mee_NJetEq4_PAS')
##histBaseNames.append('Mee_NJetEq5_PAS')
##histBaseNames.append('Mee_NJetEq6_PAS')
##histBaseNames.append('Mee_NJetEq7_PAS')
##histBaseNames.append('Mee_NJetGeq3_PAS')
##histBaseNames.append('Mee_NJetGeq4_PAS')
##histBaseNames.append('Mee_sT300To500_PAS')
##histBaseNames.append('Mee_sT500To750_PAS')
##histBaseNames.append('Mee_sT750To1250_PAS')
##histBaseNames.append('Mee_sT1250ToInf_PAS')
##histBaseNames.append('Mee_MejMin100To200_PAS')
##histBaseNames.append('Mee_MejMin200To300_PAS')
##histBaseNames.append('Mee_MejMin300To400_PAS')
##histBaseNames.append('Mee_MejMin400To500_PAS')
##histBaseNames.append('Mee_MejMin500To650_PAS')
##histBaseNames.append('Mee_MejMin650ToInf_PAS')
## for rescale factor as a function of sT
#histBaseNames.append( "Mee_sT340_PAS")
#histBaseNames.append( "Mee_sT405_PAS")
#histBaseNames.append( "Mee_sT470_PAS")
#histBaseNames.append( "Mee_sT535_PAS")
#histBaseNames.append( "Mee_sT595_PAS")
#histBaseNames.append( "Mee_sT660_PAS")
#histBaseNames.append( "Mee_sT720_PAS")
#histBaseNames.append( "Mee_sT780_PAS")
#histBaseNames.append( "Mee_sT840_PAS")
#histBaseNames.append( "Mee_sT900_PAS")
#histBaseNames.append( "Mee_sT960_PAS")
#histBaseNames.append( "Mee_sT1015_PAS")
#histBaseNames.append( "Mee_sT1075_PAS")
#histBaseNames.append( "Mee_sT1130_PAS")
#histBaseNames.append( "Mee_sT1190_PAS")
#histBaseNames.append( "Mee_sT1245_PAS")
#histBaseNames.append( "Mee_sT1300_PAS")
#histBaseNames.append( "Mee_sT1355_PAS")
#histBaseNames.append( "Mee_sT1410_PAS")
#histBaseNames.append( "Mee_sT1460_PAS")
#histBaseNames.append( "Mee_sT1515_PAS")
#histBaseNames.append( "Mee_sT1565_PAS")
#histBaseNames.append( "Mee_sT1615_PAS")
#histBaseNames.append( "Mee_sT1670_PAS")
#histBaseNames.append( "Mee_sT1720_PAS")
#histBaseNames.append( "Mee_sT1770_PAS")
#histBaseNames.append( "Mee_sT1815_PAS")

plots = []

for histBaseName in histBaseNames:
  # MG HT BKG
  h_ALLBKG_HT_Mee = GetHisto("histo1D__ALLBKG_MG_HT__"+histBaseName, File_preselection) # MC all
  # amc@NLO BG
  h_ALLBKG_amcatnlo_Mee = GetHisto("histo1D__ALLBKG_amcAtNLOIncTTBar_ZJetWJetPt__"+histBaseName, File_preselection) # MC all
  
  h_TTbarMadgraph_Mee = GetHisto("histo1D__TTbar_Madgraph__"+histBaseName, File_preselection) # MC TTbar
  h_ZJets_Mee = GetHisto("histo1D__ZJet_Madgraph_HT__"+histBaseName, File_preselection)
  h_WJets_Mee = GetHisto("histo1D__WJet_Madgraph_HT__"+histBaseName, File_preselection)
  h_TTbar_amcatnlo_Mee = GetHisto("histo1D__TTbar_amcatnlo_Inc__"+histBaseName, File_preselection) # MC TTbar
  h_ZJets_amcatnlo_Mee = GetHisto("histo1D__ZJet_amcatnlo_ptBinned__"+histBaseName, File_preselection)
  h_WJets_amcatnlo_Mee = GetHisto("histo1D__WJet_amcatnlo_ptBinned__"+histBaseName, File_preselection)
  h_SingleTop_Mee = GetHisto("histo1D__SingleTop__"+histBaseName, File_preselection)
  h_PhotonJets_Mee = GetHisto("histo1D__PhotonJets_Madgraph__"+histBaseName, File_preselection)
  h_Diboson_Mee = GetHisto("histo1D__DIBOSON__"+histBaseName, File_preselection)
  
  # DATA
  h_DATA_Mee = GetHisto("histo1D__DATA__"+histBaseName, File_preselection) #DATA
  # QCD
  #h_QCD_DataDriven = GetHisto("histo1D__QCDFakes_DATA__"+histBaseName,File_QCD_preselection)
  h_QCD = GetHisto("histo1D__QCD_EMEnriched__"+histBaseName,File_QCD_preselection)
  
  plot0 = Plot()
  plot0.histoDATA = h_DATA_Mee
  plot0.histoMCall = h_ALLBKG_amcatnlo_Mee
  plot0.histoMCTTbar = h_TTbar_amcatnlo_Mee
  plot0.histoQCD = h_QCD
  plot0.histoZJet = h_ZJets_amcatnlo_Mee
  plot0.histoWJet = h_WJets_amcatnlo_Mee
  plot0.histoSingleTop = h_SingleTop_Mee
  plot0.histoPhotonJets = h_PhotonJets_Mee
  plot0.histoDiboson = h_Diboson_Mee
  plot0.xmin = 110
  plot0.xmax = h_TTbarMadgraph_Mee.GetXaxis().GetXmax()
  #plot0.name = "TTbarRescale"
  plot0.name = histBaseName
  plot0.fileXsectionNoRescale = "/afs/cern.ch/user/s/scooper/work/private/cmssw/8011/TestRootNTuplizerRecipe/src/Leptoquarks/analyzer/rootNtupleAnalyzerV2/config/xsection_13TeV_2015.txt"
  plot0.xminplot = 0
  plot0.xmaxplot = 2000
  plot0.yminplot = 0
  plot0.ymaxplot = 2000
  plot0.datasetName = "TTJets_.+Tune"
  #plot0.datasetName = "DYJetsToLL_M-50_HT.+Tune"
  #plot0.datasetName = "Z.+Jets_Pt.+alpgen"
  # example: this match with /Z3Jets_Pt300to800-alpgen/Spring10-START3X_V26_S09-v1/GEN-SIM-RECO
  plots.append(plot0)


#-----------------------------------------------------------------------------------


############# USER CODE - END ################################################
##############################################################################



#--- Generate and print the plots from the list 'plots' define above

#--- Output files
fileps = "allPlots_calc_TTBarRescale_AND_xsecFile.ps"

#--- Generate and print the plots from the list 'plots' define above
#XXX remove printing stuff
#c = TCanvas()
#c.Print(fileps+"[")
for plot in plots:
    plot.CalculateRescaleFactor(fileps)
#c.Print(fileps+"]")
#os.system('ps2pdf '+fileps)

