#define analysisClass_cxx
#include "analysisClass.h"
#include <TH2.h>
#include <TH1F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <TVector2.h>
#include <TVector3.h>

//#include <TFile.h>
//#include <TH1F.h>
//#include <TF1.h>
//#include <TFitResultPtr.h>
//#include <TFitResult.h>
//
//#include "qcdFitter.h"
// for scale factors
#include "ElectronScaleFactors.C"
// 2016 trigger efficiency
#include "TriggerEfficiency2016.h"
// for prescales
#include "include/Run2PhotonTriggerPrescales.h"
#include "include/HistoReader.h"


analysisClass::analysisClass(string * inputList, string * cutFile, string * treeName, string * outputFileName, string * cutEfficFile)
  :baseClass(inputList, cutFile, treeName, outputFileName, cutEfficFile){}

analysisClass::~analysisClass(){}
   
void analysisClass::Loop()
{
   std::cout << "analysisClass::Loop() begins" <<std::endl;   

   //--------------------------------------------------------------------------
   // Decide which plots to save (default is to save everything)
   //--------------------------------------------------------------------------
   
   fillSkim                         (  true  ) ;
   fillAllPreviousCuts              (  true  ) ;
   fillAllOtherCuts                 ( !true  ) ;
   fillAllSameLevelAndLowerLevelCuts( !true  ) ;
   fillAllCuts                      ( !true  ) ;

   //--------------------------------------------------------------------------
   // Get pre-cut values
   //--------------------------------------------------------------------------

   // eta boundaries
   double eleEta_bar_max = getPreCutValue1("eleEta_bar");
   double eleEta_end_min = getPreCutValue1("eleEta_end1");
   double eleEta_end_max = getPreCutValue2("eleEta_end2");


   // eta boundaries
   double eleEta_bar            = getPreCutValue1("eleEta_bar");
   double eleEta_end1_min       = getPreCutValue1("eleEta_end1");
   double eleEta_end1_max       = getPreCutValue2("eleEta_end1");
   double eleEta_end2_min       = getPreCutValue1("eleEta_end2");
   double eleEta_end2_max       = getPreCutValue2("eleEta_end2");


   // override the fake rate?
   double fakeRate_override = getPreCutValue1("fakeRate_override");
   bool override_fakeRate = ( fakeRate_override > 0.0 );

   //--------------------------------------------------------------------------
   // QCD Fake Rate loading part
   //--------------------------------------------------------------------------
   std::string qcdFileName = getPreCutString1("QCDFakeRateFileName");
   HistoReader qcdFakeRateReader(qcdFileName,"fr2D_Bar_2Jet","fr2D_End_2Jet",true,false);

   // prescales
   Run2PhotonTriggerPrescales run2PhotonTriggerPrescales;

   //--------------------------------------------------------------------------
   // Analysis year
   //--------------------------------------------------------------------------
   int analysisYear = getPreCutValue1("AnalysisYear");

   //--------------------------------------------------------------------------
   // reco scale factors
   //--------------------------------------------------------------------------
   std::string recoSFFileName = getPreCutString1("RecoSFFileName");
   std::unique_ptr<HistoReader> recoScaleFactorReader = std::unique_ptr<HistoReader>(new HistoReader(recoSFFileName,"EGamma_SF2D","EGamma_SF2D",true,false));

   //--------------------------------------------------------------------------
   // Create TH1D's
   //--------------------------------------------------------------------------
   
   CreateUserTH1D( "nElectron_PAS"         ,    5   , -0.5    , 4.5      );
   CreateUserTH1D( "nMuon_PAS"             ,    5   , -0.5    , 4.5      );
   CreateUserTH1D( "nJet_PAS"              ,    10  , -0.5    , 9.5      );
   CreateUserTH1D( "Pt1stEle_PAS"	   , 	100 , 0       , 1000     ); 
   CreateUserTH1D( "Eta1stEle_PAS"	   , 	100 , -5      , 5	 ); 
   CreateUserTH1D( "Phi1stEle_PAS"	   , 	60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "Pt2ndEle_PAS"	   , 	100 , 0       , 1000     ); 
   CreateUserTH1D( "Eta2ndEle_PAS"	   , 	100 , -5      , 5	 ); 
   CreateUserTH1D( "Phi2ndEle_PAS"	   , 	60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "Charge1stEle_PAS"	   , 	2   , -1.0001 , 1.0001	 ); 
   CreateUserTH1D( "Charge2ndEle_PAS"	   , 	2   , -1.0001 , 1.0001	 ); 
   CreateUserTH1D( "MET_PAS"               ,    200 , 0       , 1000	 ); 
   CreateUserTH1D( "METPhi_PAS"		   , 	60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "Pt1stJet_PAS"          ,    100 , 0       , 1000	 ); 
   CreateUserTH1D( "Eta1stJet_PAS"         ,    100 , -5      , 5	 ); 
   CreateUserTH1D( "Phi1stJet_PAS"	   , 	60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "sT_PAS"                ,    200 , 0       , 2000	 ); 
   CreateUserTH1D( "Mee_PAS"		   ,    2000 , 0       , 2000	 ); 
   CreateUserTH1D( "Me1j1_PAS"		   ,    200 , 0       , 2000	 ); 
   CreateUserTH1D( "Me2j1_PAS"		   ,    200 , 0       , 2000	 ); 
   CreateUserTH1D( "Meejj_PAS"             ,    200 , 0       , 2000     );
   CreateUserTH1D( "Ptee_PAS"              ,    200 , 0       , 2000     );
   		                           
		                           
   CreateUserTH1D( "nVertex_PAS"           ,    31   , -0.5   , 30.5	 ) ; 
		                           
   CreateUserTH1D( "DR_Ele1Jet1_PAS"	   , 	getHistoNBins("DR_Ele1Jet1"), getHistoMin("DR_Ele1Jet1"), getHistoMax("DR_Ele1Jet1")     ) ; 
   CreateUserTH1D( "DR_Ele2Jet1_PAS"	   , 	getHistoNBins("DR_Ele2Jet1"), getHistoMin("DR_Ele2Jet1"), getHistoMax("DR_Ele2Jet1")     ) ; 


   //--------------------------------------------------------------------------
   // Tell the user how many entries we'll look at
   //--------------------------------------------------------------------------

   Long64_t nentries = GetTreeEntries();
   std::cout << "analysisClass::analysisClass(): nentries = " << nentries << std::endl;

   //--------------------------------------------------------------------------
   // Loop over the chain
   //--------------------------------------------------------------------------
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     readerTools_->LoadEntry(jentry);
     //-----------------------------------------------------------------
     // Print progress
     //-----------------------------------------------------------------
     if(jentry < 10 || jentry%5000 == 0) std::cout << "analysisClass:Loop(): jentry = " << jentry << "/" << nentries << std::endl;
     //// run ls event
     //std::cout << static_cast<unsigned int>(run) << " " << static_cast<unsigned int>(ls) << " " << static_cast<unsigned int>(event) << std::endl;

     //--------------------------------------------------------------------------
     // Reset the cuts
     //--------------------------------------------------------------------------

     resetCuts();

     //--------------------------------------------------------------------------
     // Check good run list
     //--------------------------------------------------------------------------
     
     double run = readerTools_->ReadValueBranch<Double_t>("run");
     int passedJSON = passJSON ( run,
         readerTools_->ReadValueBranch<Double_t>("ls"),
         isData() ) ;

     //--------------------------------------------------------------------------
     // Do pileup re-weighting
     //--------------------------------------------------------------------------
     
     double pileup_weight = readerTools_->ReadValueBranch<Double_t>("puWeight");
     if ( isData() ) pileup_weight = 1.0;
     
     //--------------------------------------------------------------------------
     // Get information about gen-level reweighting (should be for Sherpa only)
     //--------------------------------------------------------------------------

     double gen_weight = readerTools_->ReadValueBranch<Double_t>("Weight");
     if ( isData() ) gen_weight = 1.0;
     //if ( isData && Ele2_ValidFrac > 998. ){
     //  gen_weight = 0.0;
     //  if      (  60.0 < M_e1e2 < 120. ) gen_weight = 0.61;
     //  else if ( 120.0 < M_e1e2 < 200. ) gen_weight = 0.42;
     //  else if ( 200.0 < M_e1e2        ) gen_weight = 0.42;
     //}

     // std::cout << "Gen weight = " << int ( 1.0 / gen_weight ) << std::endl;
     //std::cout << "Gen weight = " << gen_weight << "; isData? " << isData() << std::endl;

     std::string current_file_name ( readerTools_->GetTree()->GetCurrentFile()->GetName());

     // SIC remove March 2018
     //// TopPt reweight
     //// only valid for powheg
     //if(current_file_name.find("TT_") != std::string::npos) {
     //  gen_weight*=TopPtWeight;
     //}
    
     //--------------------------------------------------------------------------
     // Electron scale factors for MC only
     //--------------------------------------------------------------------------
     if(!isData()) {
       float ele1PtUncorr = readerTools_->ReadValueBranch<Double_t>("LooseEle1_Pt");// loose ele Pt is now uncorrected /readerTools_->ReadValueBranch<Double_t>("LooseEle1_ECorr");
       float ele2PtUncorr = readerTools_->ReadValueBranch<Double_t>("LooseEle2_Pt");// loose ele Pt is now uncorrected /readerTools_->ReadValueBranch<Double_t>("LooseEle2_ECorr");
       //float ele1PtUncorr = readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEt");
       //float ele2PtUncorr = readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEt");
       //std::cout << "LooseEle1_Pt = " << readerTools_->ReadValueBranch<Double_t>("LooseEle1_Pt") << "; LooseEle1_ECorr = " << ele1ECorr << "; LooseEle1_SCEta = " << readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta") << std::endl;
       //std::cout << "LooseEle2_Pt = " << readerTools_->ReadValueBranch<Double_t>("LooseEle2_Pt") << "; LooseEle2_ECorr = " << ele2ECorr << "; LooseEle2_SCEta = " << readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta") << std::endl;
       float recoHeepSF = 1.0;
       bool verbose = false;
       if(analysisYear==2016) {
         float recoSFLooseEle1 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"),ele1PtUncorr,verbose);
         float recoSFLooseEle2 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"),ele2PtUncorr,verbose);
         float heepSFLooseEle1 = ElectronScaleFactors2016::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"));
         float heepSFLooseEle2 = ElectronScaleFactors2016::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"));
         recoHeepSF *= recoSFLooseEle1*recoSFLooseEle2*heepSFLooseEle1*heepSFLooseEle2;
       }
       else if(analysisYear==2017) {
         float zVtxSF = ElectronScaleFactors2017::zVtxSF;
         float recoSFLooseEle1 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"),ele1PtUncorr,verbose);
         float recoSFLooseEle2 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"),ele2PtUncorr,verbose);
         float heepSFLooseEle1 = ElectronScaleFactors2017::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"));
         float heepSFLooseEle2 = ElectronScaleFactors2017::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"));
         recoHeepSF *= zVtxSF*recoSFLooseEle1*recoSFLooseEle2*heepSFLooseEle1*heepSFLooseEle2;
       }
       else if(analysisYear==2018) {
         float recoSFLooseEle1 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"),ele1PtUncorr,verbose);
         float recoSFLooseEle2 = recoScaleFactorReader->LookupValue(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"),ele2PtUncorr,verbose);
         float heepSFLooseEle1 = ElectronScaleFactors2018::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta"));
         float heepSFLooseEle2 = ElectronScaleFactors2018::LookupHeepSF(readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta"));
         recoHeepSF *= recoSFLooseEle1*recoSFLooseEle2*heepSFLooseEle1*heepSFLooseEle2;
       }
       //FIXME: in the case that one ele passes HEEP, need to apply that SF only
       // we only care about MC for cej, 1 pass HEEP and 1 fail HEEP
       gen_weight*=recoHeepSF;
       // stick the gen_weight in with the pileup_weight
       pileup_weight*=gen_weight;
     }

     //--------------------------------------------------------------------------
     // Trigger
     //--------------------------------------------------------------------------
     // Find the right prescale for this event
     double min_prescale = 1;
     int passTrigger = 0;
     std::string triggerName = "";

     double LooseEle1_hltPhotonPt = readerTools_->ReadValueBranch<Double_t>("LooseEle1_hltPhotonPt");

     if ( LooseEle1_hltPhotonPt > 0.0 ) {
       if(analysisYear==2016) {
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon22")   > 0.1 && LooseEle1_hltPhotonPt >= 22.  && LooseEle1_hltPhotonPt < 30. ) { passTrigger = 1; triggerName = "Photon22"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon30")   > 0.1 && LooseEle1_hltPhotonPt >= 30.  && LooseEle1_hltPhotonPt < 36. ) { passTrigger = 1; triggerName = "Photon30"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon36")   > 0.1 && LooseEle1_hltPhotonPt >= 36.  && LooseEle1_hltPhotonPt < 50. ) { passTrigger = 1; triggerName = "Photon36"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon50")   > 0.1 && LooseEle1_hltPhotonPt >= 50.  && LooseEle1_hltPhotonPt < 75. ) { passTrigger = 1; triggerName = "Photon50"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon75")   > 0.1 && LooseEle1_hltPhotonPt >= 75.  && LooseEle1_hltPhotonPt < 90. ) { passTrigger = 1; triggerName = "Photon75"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon90")   > 0.1 && LooseEle1_hltPhotonPt >= 90.  && LooseEle1_hltPhotonPt < 120.) { passTrigger = 1; triggerName = "Photon90"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon120")  > 0.1 && LooseEle1_hltPhotonPt >= 120. && LooseEle1_hltPhotonPt < 175.) { passTrigger = 1; triggerName = "Photon120"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon175")  > 0.1 && LooseEle1_hltPhotonPt >= 175.) { passTrigger = 1; triggerName = "Photon175"; } 
       }
       else if(analysisYear==2017) {
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon25")   > 0.1 && LooseEle1_hltPhotonPt >= 25.  && LooseEle1_hltPhotonPt < 33. ) { passTrigger = 1; triggerName = "Photon25"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon33")   > 0.1 && LooseEle1_hltPhotonPt >= 33.  && LooseEle1_hltPhotonPt < 50. ) { passTrigger = 1; triggerName = "Photon33"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon50")   > 0.1 && LooseEle1_hltPhotonPt >= 50.  && LooseEle1_hltPhotonPt < 75. ) { passTrigger = 1; triggerName = "Photon50"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon75")   > 0.1 && LooseEle1_hltPhotonPt >= 75.  && LooseEle1_hltPhotonPt < 90. ) { passTrigger = 1; triggerName = "Photon75"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon90")   > 0.1 && LooseEle1_hltPhotonPt >= 90.  && LooseEle1_hltPhotonPt < 120.) { passTrigger = 1; triggerName = "Photon90"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon120")  > 0.1 && LooseEle1_hltPhotonPt >= 120. && LooseEle1_hltPhotonPt < 150.) { passTrigger = 1; triggerName = "Photon120"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon150")  > 0.1 && LooseEle1_hltPhotonPt >= 150. && LooseEle1_hltPhotonPt < 175.) { passTrigger = 1; triggerName = "Photon150"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon175")  > 0.1 && LooseEle1_hltPhotonPt >= 175. && LooseEle1_hltPhotonPt < 200.) { passTrigger = 1; triggerName = "Photon175"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon200")  > 0.1 && LooseEle1_hltPhotonPt >= 200.) { passTrigger = 1; triggerName = "Photon200"; } 
       }
       else if(analysisYear==2018) {
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon33")   > 0.1 && LooseEle1_hltPhotonPt >= 33.  && LooseEle1_hltPhotonPt < 50. ) { passTrigger = 1; triggerName = "Photon33"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon50")   > 0.1 && LooseEle1_hltPhotonPt >= 50.  && LooseEle1_hltPhotonPt < 75. ) { passTrigger = 1; triggerName = "Photon50"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon75")   > 0.1 && LooseEle1_hltPhotonPt >= 75.  && LooseEle1_hltPhotonPt < 90. ) { passTrigger = 1; triggerName = "Photon75"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon90")   > 0.1 && LooseEle1_hltPhotonPt >= 90.  && LooseEle1_hltPhotonPt < 120.) { passTrigger = 1; triggerName = "Photon90"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon120")  > 0.1 && LooseEle1_hltPhotonPt >= 120. && LooseEle1_hltPhotonPt < 150.) { passTrigger = 1; triggerName = "Photon120"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon150")  > 0.1 && LooseEle1_hltPhotonPt >= 150. && LooseEle1_hltPhotonPt < 175.) { passTrigger = 1; triggerName = "Photon150"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon175")  > 0.1 && LooseEle1_hltPhotonPt >= 175. && LooseEle1_hltPhotonPt < 200.) { passTrigger = 1; triggerName = "Photon175"; } 
         if ( readerTools_->ReadValueBranch<Double_t>("H_Photon200")  > 0.1 && LooseEle1_hltPhotonPt >= 200.) { passTrigger = 1; triggerName = "Photon200"; } 
       }
     }
     if(isData() && passTrigger) {
       //std::cout << "INFO: lookup trigger name " << triggerName << " for year: " << year << std::endl;
       min_prescale = run2PhotonTriggerPrescales.LookupPrescale(analysisYear,triggerName);
     }


     //--------------------------------------------------------------------------
     // What kind of event is this?
     //   - Barrel
     //   - Endcap 1 (eta < 2.0)
     //   - Endcap 2 (eta > 2.0) 
     //--------------------------------------------------------------------------

     bool ele1_isBarrel  = false;
     bool ele1_isEndcap1 = false;
     bool ele1_isEndcap2 = false;
     bool ele2_isBarrel  = false;
     bool ele2_isEndcap1 = false;
     bool ele2_isEndcap2 = false;

     double LooseEle1_SCEta = readerTools_->ReadValueBranch<Double_t>("LooseEle1_SCEta");
     double LooseEle2_SCEta = readerTools_->ReadValueBranch<Double_t>("LooseEle2_SCEta");
     if( fabs( LooseEle1_SCEta  ) < eleEta_bar )        ele1_isBarrel  = true;
     if( fabs( LooseEle1_SCEta  ) > eleEta_end1_min &&
         fabs( LooseEle1_SCEta  ) < eleEta_end1_max )   ele1_isEndcap1 = true;
     if( fabs( LooseEle1_SCEta  ) > eleEta_end2_min &&
         fabs( LooseEle1_SCEta  ) < eleEta_end2_max )   ele1_isEndcap2 = true;

     if( fabs( LooseEle2_SCEta  ) < eleEta_bar )        ele2_isBarrel  = true;
     if( fabs( LooseEle2_SCEta  ) > eleEta_end1_min &&
         fabs( LooseEle2_SCEta  ) < eleEta_end1_max )   ele2_isEndcap1 = true;
     if( fabs( LooseEle2_SCEta  ) > eleEta_end2_min &&
         fabs( LooseEle2_SCEta  ) < eleEta_end2_max )   ele2_isEndcap2 = true;

     bool ele1_isEndcap = ( ele1_isEndcap1 || ele1_isEndcap2 ) ;
     bool ele2_isEndcap = ( ele2_isEndcap1 || ele2_isEndcap2 ) ;

     bool isEBEB = ( ele1_isBarrel && ele2_isBarrel ) ;
     bool isEBEE = ( ( ele1_isBarrel && ele2_isEndcap ) ||
         ( ele2_isBarrel && ele1_isEndcap ) );
     bool isEEEE = ( ele1_isEndcap && ele2_isEndcap ) ;
     bool isEB   = ( isEBEB || isEBEE ) ;

     //--------------------------------------------------------------------------
     // Make this a QCD fake rate calculation
     //--------------------------------------------------------------------------
     // LooseEle Pt is the uncorrected SCEt
     double LooseEle1_Pt = readerTools_->ReadValueBranch<Double_t>("LooseEle1_Pt");// loose ele Pt is now uncorrected /readerTools_->ReadValueBranch<Double_t>("LooseEle1_ECorr");
     double LooseEle2_Pt = readerTools_->ReadValueBranch<Double_t>("LooseEle2_Pt");// loose ele Pt is now uncorrected /readerTools_->ReadValueBranch<Double_t>("LooseEle2_ECorr");
     bool verboseFakeRateCalc = false;
     float fakeRate1 = qcdFakeRateReader.LookupValue(LooseEle1_SCEta,LooseEle1_Pt,verboseFakeRateCalc);
     float fakeRate2 = qcdFakeRateReader.LookupValue(LooseEle2_SCEta,LooseEle2_Pt,verboseFakeRateCalc);

     //--------------------------------------------------------------------------
     // Finally have the effective fake rate
     //--------------------------------------------------------------------------

     //FIXME: add error on fake rate as well
     double fakeRateEffective  = fakeRate1/(1-fakeRate1); // require loose electron to fail HEEP ID
     //if(1-fakeRate1 <= 0)
     //{
     //  cout << "ERROR: Found fakeRate1: " << fakeRate1 << " for SCEta=" << LooseEle1_SCEta << " SCEt="
     //    << LooseEle1_SCEnergy/cosh(LooseEle1_SCEta) << "=" << LooseEle1_SCEnergy << "/" << 
     //    cosh(LooseEle1_SCEta) << endl;
     //}
     double nLooseEle_store = readerTools_->ReadValueBranch<Double_t>("nLooseEle_store");
     if ( nLooseEle_store >= 2 ) { 							        
       fakeRateEffective += fakeRate2/(1-fakeRate2);
     }
     // double eFakeRateEffective = fakeRateEffective * sqrt (  ( eFakeRate1 / fakeRate1 ) * ( eFakeRate1 / fakeRate1 ) +
     //					     ( eFakeRate2 / fakeRate2 ) * ( eFakeRate2 / fakeRate2 ) );
     double eFakeRateEffective = 0.0;

     //--------------------------------------------------------------------------
     // User has the option to use a flat fake rate (e.g. 1.0 = no fake rate)
     //--------------------------------------------------------------------------
     
     if ( override_fakeRate ) fakeRateEffective = fakeRate_override;

     //--------------------------------------------------------------------------
     // How many loose electrons have HEEP ID?
     //--------------------------------------------------------------------------

     int nPass = 0;
     if ( readerTools_->ReadValueBranch<Double_t>("LooseEle1_PassHEEPID") == 1 ) nPass ++;
     if ( readerTools_->ReadValueBranch<Double_t>("LooseEle2_PassHEEPID") == 1 ) nPass ++;

     //--------------------------------------------------------------------------
     // Calculate a few missing variables
     //--------------------------------------------------------------------------

     double LooseEle1_Eta = readerTools_->ReadValueBranch<Double_t>("LooseEle1_Eta");
     double LooseEle2_Eta = readerTools_->ReadValueBranch<Double_t>("LooseEle2_Eta");
     double LooseEle1_Phi = readerTools_->ReadValueBranch<Double_t>("LooseEle1_Phi");
     double LooseEle2_Phi = readerTools_->ReadValueBranch<Double_t>("LooseEle2_Phi");
     double LooseEle1_Charge = readerTools_->ReadValueBranch<Double_t>("LooseEle1_Charge");
     double LooseEle2_Charge = readerTools_->ReadValueBranch<Double_t>("LooseEle2_Charge");
     double JetLooseEle1_Pt = readerTools_->ReadValueBranch<Double_t>("JetLooseEle1_Pt");
     double JetLooseEle2_Pt = readerTools_->ReadValueBranch<Double_t>("JetLooseEle2_Pt");
     double JetLooseEle1_Eta = readerTools_->ReadValueBranch<Double_t>("JetLooseEle1_Eta");
     double JetLooseEle2_Eta = readerTools_->ReadValueBranch<Double_t>("JetLooseEle2_Eta");
     double JetLooseEle1_Phi = readerTools_->ReadValueBranch<Double_t>("JetLooseEle1_Phi");
     double JetLooseEle2_Phi = readerTools_->ReadValueBranch<Double_t>("JetLooseEle2_Phi");
     double nMuon_ptCut = readerTools_->ReadValueBranch<Double_t>("nMuon_ptCut");
     TLorentzVector loose_ele1, loose_ele2 , jet1, jet2;
     loose_ele1.SetPtEtaPhiM ( LooseEle1_Pt , LooseEle1_Eta , LooseEle1_Phi , 0.0 );
     loose_ele2.SetPtEtaPhiM ( LooseEle2_Pt , LooseEle2_Eta , LooseEle2_Phi , 0.0 );
     jet1.SetPtEtaPhiM       ( JetLooseEle1_Pt, JetLooseEle1_Eta, JetLooseEle1_Phi, 0.0 );
     jet2.SetPtEtaPhiM       ( JetLooseEle2_Pt, JetLooseEle2_Eta, JetLooseEle2_Phi, 0.0 );

     TLorentzVector loose_e1e2 = loose_ele1 + loose_ele2;
     TLorentzVector j1j2 = jet1 + jet2;

     TLorentzVector e1j1 = loose_ele1 + jet1;
     TLorentzVector e2j1 = loose_ele2 + jet1;

     //--------------------------------------------------------------------------
     // Now fill cut values
     // DON'T use the pileup weight ... it's included by default
     // DO    use the prescale.  It's already 1.0 for MC.
     // DO    use the effective fake rate.  It will only mean anything when you run
     //       over data, though, so be sure to override it to 1.0 
     //       if you run over Monte Carlo
     //--------------------------------------------------------------------------


     //--------------------------------------------------------------------------
     // Fill variables
     //--------------------------------------------------------------------------
     double PFMET_Type1_Pt  = readerTools_->ReadValueBranch<Double_t>("PFMET_Type1_Pt");
     double PFMET_Type1_Phi  = readerTools_->ReadValueBranch<Double_t>("PFMET_Type1_Phi");
     double nLooseEle_ptCut = readerTools_->ReadValueBranch<Double_t>("nLooseEle_ptCut");
     double nJetLooseEle_ptCut = readerTools_->ReadValueBranch<Double_t>("nJetLooseEle_ptCut");
     double nJetLooseEle_store = readerTools_->ReadValueBranch<Double_t>("nJetLooseEle_store");
     double nVertex = readerTools_->ReadValueBranch<Double_t>("nVertex");
     // reweighting
     fillVariableWithValue ( "Reweighting", 1, pileup_weight * min_prescale * fakeRateEffective) ; 

     // JSON variable
     fillVariableWithValue( "PassJSON" , passedJSON, pileup_weight * min_prescale * fakeRateEffective) ; 

     fillVariableWithValue ( "PassHLT", passTrigger, pileup_weight * min_prescale * fakeRateEffective ) ;

     // Fill noise filters
     // see: https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
     // we filled these at skim time
     fillVariableWithValue("PassGlobalSuperTightHalo2016Filter" , int(readerTools_->ReadValueBranch<Double_t>("PassGlobalSuperTightHalo2016Filter")     == 1), fakeRateEffective * min_prescale);
     fillVariableWithValue("PassGoodVertices"                   , int(readerTools_->ReadValueBranch<Double_t>("PassGoodVertices")                       == 1), fakeRateEffective * min_prescale);
     fillVariableWithValue("PassHBHENoiseFilter"                , int(readerTools_->ReadValueBranch<Double_t>("PassHBHENoiseFilter")                    == 1), fakeRateEffective * min_prescale);
     fillVariableWithValue("PassHBHENoiseIsoFilter"             , int(readerTools_->ReadValueBranch<Double_t>("PassHBHENoiseIsoFilter")                 == 1), fakeRateEffective * min_prescale);
     // eBadScFilter not suggested for MC
     if(isData())
       fillVariableWithValue("PassBadEESupercrystalFilter"      , int(readerTools_->ReadValueBranch<Double_t>("PassBadEESupercrystalFilter")            == 1), fakeRateEffective * min_prescale);
     else
       fillVariableWithValue("PassBadEESupercrystalFilter"      , 1                                                                                          , fakeRateEffective * min_prescale);
     fillVariableWithValue("PassEcalDeadCellTrigPrim"           , int(readerTools_->ReadValueBranch<Double_t>("PassEcalDeadCellTrigPrim")               == 1), fakeRateEffective * min_prescale);
     // not recommended
     //fillVariableWithValue("PassChargedCandidateFilter"         , int(readerTools_->ReadValueBranch<Double_t>("PassChargedCandidateFilter")             == 1), fakeRateEffective * min_prescale);
     fillVariableWithValue("PassBadPFMuonFilter"                , int(readerTools_->ReadValueBranch<Double_t>("PassBadPFMuonFilter")                    == 1), fakeRateEffective * min_prescale);
     // EcalBadCalibV2 for 2017, 2018
     if(analysisYear > 2016)
       fillVariableWithValue("PassEcalBadCalibV2Filter"         , int(readerTools_->ReadValueBranch<Double_t>("PassEcalBadCalibV2Filter")               == 1), fakeRateEffective * min_prescale);
     else
       fillVariableWithValue("PassEcalBadCalibV2Filter"         , 1                                                                                          , fakeRateEffective * min_prescale);

     // MET
     fillVariableWithValue ( "PFMET"  , PFMET_Type1_Pt, pileup_weight * min_prescale * fakeRateEffective ) ;
     
     // Muons
     fillVariableWithValue(   "nMuon"                         , nMuon_ptCut           , pileup_weight * min_prescale  * fakeRateEffective ); 
										      			      
     // Electrons
     fillVariableWithValue(   "nEleLoose"                     , nLooseEle_ptCut     , pileup_weight * min_prescale  * fakeRateEffective );
     fillVariableWithValue(   "nEleTight"                     , nPass            , pileup_weight * min_prescale  * fakeRateEffective );
     if ( nLooseEle_ptCut >= 1 ) { 
       fillVariableWithValue( "Ele1_Pt"                       , LooseEle1_Pt   , pileup_weight * min_prescale  * fakeRateEffective ) ;
       fillVariableWithValue( "Ele1_Eta"                      , LooseEle1_Eta  , pileup_weight * min_prescale  * fakeRateEffective ) ;
     }
     if ( nLooseEle_ptCut >= 2 ) { 
       fillVariableWithValue( "Ele2_Pt"                       , LooseEle2_Pt   , pileup_weight * min_prescale  * fakeRateEffective ) ;
       fillVariableWithValue( "Ele2_Eta"                      , LooseEle2_Eta  , pileup_weight * min_prescale  * fakeRateEffective ) ;
       fillVariableWithValue( "M_e1e2"                        , loose_e1e2.M()   , pileup_weight * min_prescale  * fakeRateEffective );
       fillVariableWithValue( "Pt_e1e2"                       , loose_e1e2.Pt()  , pileup_weight * min_prescale  * fakeRateEffective );
     }

     // Jets
     fillVariableWithValue(   "nJet"                          , nJetLooseEle_ptCut , pileup_weight * min_prescale  * fakeRateEffective );
     if ( nJetLooseEle_store >= 1 ) {
       fillVariableWithValue( "Jet1_Pt"                       , JetLooseEle1_Pt   , pileup_weight * min_prescale  * fakeRateEffective ) ;
       fillVariableWithValue( "Jet1_Eta"                      , JetLooseEle1_Eta , pileup_weight * min_prescale  * fakeRateEffective ) ;
     }

     // DeltaR
     if ( nLooseEle_ptCut >= 2 && nJetLooseEle_store >= 1) {
       double sT_eej = LooseEle1_Pt + LooseEle2_Pt + JetLooseEle1_Pt ;
       fillVariableWithValue( "DR_Ele1Jet1"                   , loose_ele1.DeltaR ( jet1 ), pileup_weight * min_prescale  * fakeRateEffective );
       fillVariableWithValue( "DR_Ele2Jet1"                   , loose_ele2.DeltaR ( jet1 ), pileup_weight * min_prescale  * fakeRateEffective );
       fillVariableWithValue( "sT_eej_200"               , sT_eej                   , pileup_weight * min_prescale  * fakeRateEffective );
       fillVariableWithValue( "sT_eej_450"               , sT_eej                   , pileup_weight * min_prescale  * fakeRateEffective );
       fillVariableWithValue( "sT_eej_850"               , sT_eej                   , pileup_weight * min_prescale  * fakeRateEffective );
     }      

     //--------------------------------------------------------------------------
     // Evaluate the cuts
     //--------------------------------------------------------------------------
     
     evaluateCuts();

     //--------------------------------------------------------------------------
     // Fill preselection plots
     // DO    use the pileup weight.  It's equal to 1.0 for data.  
     // DO    use the min_prescale.  It's equal to 1.0 for Monte Carlo
     // DO    use the effective fake rate.  It will only mean anything when you run
     //       over data, though, so be sure to override it to 1.0 
     //       if you run over Monte Carlo
     //--------------------------------------------------------------------------     

     bool passed_preselection = ( passedAllPreviousCuts("sT_eej_200") && passedCut ("sT_eej_200") );
     
     if ( passed_preselection ) {

       double sT_eej = LooseEle1_Pt + LooseEle2_Pt + JetLooseEle1_Pt ;

       FillUserTH1D("nElectron_PAS"        , nLooseEle_ptCut           , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("nMuon_PAS"            , nMuon_ptCut               , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("nJet_PAS"             , nJetLooseEle_ptCut        , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Pt1stEle_PAS"	   , LooseEle1_Pt              , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Eta1stEle_PAS"	   , LooseEle1_Eta             , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Phi1stEle_PAS"	   , LooseEle1_Phi             , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Pt2ndEle_PAS"	   , LooseEle2_Pt              , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Eta2ndEle_PAS"	   , LooseEle2_Eta             , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Phi2ndEle_PAS"	   , LooseEle2_Phi             , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Charge1stEle_PAS"	   , LooseEle1_Charge          , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Charge2ndEle_PAS"	   , LooseEle2_Charge          , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("MET_PAS"              , PFMET_Type1_Pt         , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("METPhi_PAS"	         , PFMET_Type1_Phi        , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Pt1stJet_PAS"         , JetLooseEle1_Pt           , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Eta1stJet_PAS"        , JetLooseEle1_Eta          , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Phi1stJet_PAS"	   , JetLooseEle1_Phi          , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("sT_PAS"               , sT_eej                    , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Mee_PAS"		   , loose_e1e2.M()            , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Ptee_PAS"             , loose_e1e2.Pt()           , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("nVertex_PAS"          , nVertex                   , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("DR_Ele1Jet1_PAS"	   , loose_ele1.DeltaR ( jet1 ), pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("DR_Ele2Jet1_PAS"	   , loose_ele2.DeltaR ( jet1 ), pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Me1j1_PAS"            , e1j1.M()                  , pileup_weight * min_prescale * fakeRateEffective );
       FillUserTH1D("Me2j1_PAS"            , e2j1.M()                  , pileup_weight * min_prescale * fakeRateEffective );
     }
   } // End loop over events

   std::cout << "analysisClass::Loop() ends" <<std::endl;   
}

