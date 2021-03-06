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
    histoDATA   = "" # DATA
    histoMCZ    = "" # MCZ
    histoMCall  = "" # MCall
    histoQCD    = "" # QCD
    xtit        = "" # xtitle
    ytit        = "" # ytitle
    xmin        = "" # set xmin to calculate rescaling factor (-- please take into account the bin size of histograms --)
    xmax        = "" # # set xmax to calculate rescaling factor (-- please take into account the bin size of histograms --)
    xminplot    = "" # min x axis range (need to set both min and max. Leave it as is for full range)
    xmaxplot    = "" # max x axis range (need to set both min and max. Leave it as is for full range)
    yminplot    = "" # min y axis range (need to set both min and max. Leave it as is for full range)
    ymaxplot    = "" # max y axis range (need to set both min and max. Leave it as is for full range)
    lpos        = "" # legend position (default = top-right, option="bottom-center", "top-left")
    #    xlog        = "" # log scale of X axis (default = no, option="yes") ### IT SEEMS IT DOES NOT WORK
    ylog        = "" # log scale of Y axis (default = no, option="yes")
    #rebin      = "" # rebin x axis (default = 1, option = set it to whatever you want )
    name        = "" # name of the final plots
    lint        = "10.9 pb^{-1}" # integrated luminosity of the sample ( example "10 pb^{-1}" )
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
        integralMCZ = GetIntegralTH1(self.histoMCZ,self.xmin,self.xmax)
        ERRintegralMCZ = GetErrorIntegralTH1(self.histoMCZ,self.xmin,self.xmax)
        integralQCD = GetIntegralTH1(self.histoQCD,self.xmin,self.xmax)
        ERRintegralQCD = GetErrorIntegralTH1(self.histoQCD,self.xmin,self.xmax)

        #contamination from other backgrounds (except Z) in the integral range
        integralMCothers = integralMCall - integralMCZ + integralQCD
        ERRintegralMCothers = math.sqrt(ERRintegralMCall**2 + ERRintegralMCZ**2 + ERRintegralQCD**2)
        contamination = integralMCothers / integralMCall

        #DATA corrected for other bkg contamination --> best estimate of DATA (due to Z only)
        integralDATAcorr = (integralDATA - integralMCothers)
        ERRintegralDATAcorr = math.sqrt(ERRintegralDATA**2 + ERRintegralMCothers**2)

        #rescale factor
        rescale = integralDATAcorr / integralMCZ
        relERRintegralDATAcorr = ERRintegralDATAcorr / integralDATAcorr
        relERRintegralMCZ = ERRintegralMCZ / integralMCZ
        relERRrescale = math.sqrt(relERRintegralDATAcorr**2 + relERRintegralMCZ**2)

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

        #printout
        print " "
        print "######################################## "
        print "integral range:  " + str(self.xmin) + " < Mee < " + str(self.xmax) + " GeV/c2"
        print "integral MC All: "   + str( integralMCall ) + " +/- " + str( ERRintegralMCall )
        print "integral QCD:    "   + str( integralQCD ) + " +/- " + str( ERRintegralQCD )
        print "integral MC Z:   "   + str( integralMCZ ) + " +/- " + str( ERRintegralMCZ )
        print "integral DATA:   "   + str( integralDATA ) + " +/- " + str( ERRintegralDATA )
        print "contribution from other bkgs (except Z+jet): " + str(contamination*100) + "%"
        print "integral DATA (corrected for contribution from other bkgs): "  + str( integralDATAcorr ) + " +/- " + str( ERRintegralDATAcorr )
        print "rescale factor for Z background: " + str(rescale) + " +\- " + str(relERRrescale*rescale)
        print "systematical uncertainty of Z+jet background modeling: " + str(relERRrescale*100) + "%"
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
        print "New xsection file (after Z rescaling) is: " + newFileName
        print " "


############# DON'T NEED TO MODIFY ANYTHING HERE - END #######################
##############################################################################


##############################################################################
############# USER CODE - BEGING #############################################

#--- Input files
#preselection
#File_preselection = GetFile("/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/RunII/eejj_ele27wplooseData_eles35GeV_noJets_noSt_28feb2016_v1-5-3/output_cutTable_lq_eejj_loosenEleRequirements_noJetRequirement/analysisClass_lq_eejj_noJets_plots.root")
#File_preselection = GetFile("/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/RunII/eejj_1mar2016_v1-5-3/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots.root")
#File_preselection = GetFile("/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/RunII/eejj_ele27WPLooseWithZPrimeEta2p1TurnOn_13mar2016_v1-5-3/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots.root")
#File_preselection = GetFile("/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/RunII/eejj_loosenEles_2LowPTJets_ele27WPLooseWithZPrimeEta2p1TurnOn_14mar2016_v1-5-3/output_cutTable_lq_eejj_loosenEleRequirements_2lowPtJets/analysisClass_lq_eejj_plots.root")
#File_preselection = GetFile("/afs/cern.ch/user/s/scooper/work/private/data/Leptoquarks/RunII/eejj_eles50GeV_jets50GeV_sT300GeV_ele27WPLooseWithZPrimeEta2p1TurnOn_17mar2016_v1-5-3/output_cutTable_lq_eejj_eles50GeV_jets50GeV_st300GeV/analysisClass_lq_eejj_plots.root")

File_preselection = GetFile("$LQDATA/RunII/eejj_analysis_ttbarRescaleFinalSels_2jun2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_plots.root")
File_QCD_preselection = GetFile("$LQDATA/RunII/eejj_analysis_ttbarRescaleFinalSels_2jun2016/output_cutTable_lq_eejj/analysisClass_lq_eejj_QCD_plots.root")


#--- Rescaling of Z/gamma + jet background

#-----------------------------------------
h_DATA_Mee = GetHisto("histo1D__DATA__Mee_80_100_Preselection", File_preselection) #DATA
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
# MG HT
h_ALLBKG_Mee = GetHisto("histo1D__ALLBKG_MG_HT__Mee_80_100_Preselection", File_preselection) # MC all
h_ZJetMadgraph_Mee = GetHisto("histo1D__ZJet_Madgraph_HT__Mee_80_100_Preselection", File_preselection) # MC Z

# QCD
h_QCD_DataDriven = GetHisto("histo1D__QCDFakes_DATA__Mee_80_100_Preselection",File_QCD_preselection)

plot0 = Plot()
plot0.histoDATA = h_DATA_Mee
plot0.histoMCall = h_ALLBKG_Mee
plot0.histoMCZ = h_ZJetMadgraph_Mee
plot0.histoQCD = h_QCD_DataDriven
plot0.xmin = 80
plot0.xmax = 100
plot0.name = "Zrescale"
plot0.fileXsectionNoRescale = "/afs/cern.ch/user/s/scooper/work/private/cmssw/LQRootTuples7414/src/Leptoquarks/analyzer/rootNtupleAnalyzerV2/config/xsection_13TeV_2015.txt"
plot0.xminplot = 0
plot0.xmaxplot = 200
plot0.yminplot = 0
plot0.ymaxplot = 200 
plot0.datasetName = "DYJetsToLL_M-50_HT.+Tune"
#plot0.datasetName = "Z.+Jets_Pt.+alpgen"
# example: this match with /Z3Jets_Pt300to800-alpgen/Spring10-START3X_V26_S09-v1/GEN-SIM-RECO

plots = [plot0]

#-----------------------------------------------------------------------------------


############# USER CODE - END ################################################
##############################################################################



#--- Generate and print the plots from the list 'plots' define above

#--- Output files
fileps = "allPlots_calc_MCrescale_AND_xsecFile.ps"

#--- Generate and print the plots from the list 'plots' define above
c = TCanvas()
c.Print(fileps+"[")
for plot in plots:
    plot.CalculateRescaleFactor(fileps)
c.Print(fileps+"]")
os.system('ps2pdf '+fileps)

