#!/usr/bin/env python3
import math
import ROOT as r

# year = "2016postVFP"
year = "2018"
tfile = r.TFile.Open("shapeHistos_{}.root".format(year))

can = r.TCanvas()
can.SetLogy()

hist = tfile.Get("yieldVsMLQ_SingleTop")
hist.SetTitle("yieldVsMLQ for SingleTop, {}".format(year))
hist.SetStats(1)
hist.SetMarkerColor(9)
hist.SetLineColor(9)
hist.SetLineWidth(2)
hist.SetMarkerSize(0.8)
hist.SetMarkerStyle(8)
hist.SetMinimum(1e-8)
hist.SetMaximum(1e4)
hist.Draw()

func = r.TF1("func", "expo(0)", 0, 5000)
constTerm = 3.90306
constTermErr = 1.25629
slopeTerm = -0.00491907
slopeTermErr = 0.00170827
# simple lumi scaling
if year == "2016preVFP":
    scaleFactor = 0.1416840016416827
elif year == "2016postVFP":
    scaleFactor = 0.1221676839055999
elif year == "2017":
    scaleFactor = 0.3014043829098545
elif year == "2018":
    scaleFactor = 0.4347439315428627
else:
    raise RuntimeError("Couldn't understand year={}, so couldn't scale the fit function.".format(year))

expConstant = math.log(scaleFactor*math.exp(constTerm))
func.SetParameters(expConstant, slopeTerm)
func.SetLineColor(r.kBlack)
func.SetLineWidth(2)
func.Draw("same")

leg = r.TLegend()
colors = [r.kAzure, r.kAzure, r.kSpring-6, r.kSpring-6, r.kRed+1, r.kOrange+8, r.kOrange+8, r.kRed+1]
errorFuncs = []
for i in range(1, 9):
    if i == 1:
        legEntry = "constTermOnly"
        toAddConst = constTermErr
        toAddSlope = 0
    elif i == 2:
        legEntry = "constTermOnly"
        toAddConst = -1*constTermErr
        toAddSlope = 0
    elif i == 3:
        legEntry = "slopeTermOnly"
        toAddConst = 0
        toAddSlope = slopeTermErr
    elif i == 4:
        legEntry = "slopeTermOnly"
        toAddConst = 0
        toAddSlope = -1*slopeTermErr
    elif i == 5:
        legEntry = "constAndSlopeUp"
        toAddConst = constTermErr
        toAddSlope = slopeTermErr
    elif i == 6:
        legEntry = "constUpSlopeDn"
        toAddConst = constTermErr
        toAddSlope = -1*slopeTermErr
    elif i == 7:
        legEntry = "constDnSlopeUp"
        toAddConst = -1*constTermErr
        toAddSlope = slopeTermErr
    elif i == 8:
        legEntry = "constAndSlopeDown"
        toAddConst = -1*constTermErr
        toAddSlope = -1*slopeTermErr
    expConstant = math.log(scaleFactor*math.exp(constTerm+toAddConst))
    newFunc = func.Clone("newFunc"+str(i))
    newFunc.SetParameters(expConstant, slopeTerm+toAddSlope)
    newFunc.SetLineColor(colors[i-1])
    newFunc.SetLineWidth(2)
    newFunc.Draw("same")
    errorFuncs.append(newFunc)
    leg.AddEntry(newFunc, legEntry, "l")

leg.SetBorderSize(0)
leg.Draw()

r.gPad.Modified()
r.gPad.Update()

# tfile.Close()
