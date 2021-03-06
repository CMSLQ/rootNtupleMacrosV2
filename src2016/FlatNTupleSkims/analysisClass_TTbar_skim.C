#define analysisClass_cxx
#include "analysisClass.h"
#include <TH2.h>
#include <TH1F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <TVector2.h>
#include <TVector3.h>
// for trigger turn-on
#include "Ele27WPLooseTrigTurnOn.C"

analysisClass::analysisClass(string * inputList, string * cutFile, string * treeName, string * outputFileName, string * cutEfficFile)
  :baseClass(inputList, cutFile, treeName, outputFileName, cutEfficFile)
{
  std::cout << "analysisClass::analysisClass(): begins " << std::endl;

  std::cout << "analysisClass::analysisClass(): ends " << std::endl;
}

analysisClass::~analysisClass()
{
  std::cout << "analysisClass::~analysisClass(): begins " << std::endl;

  std::cout << "analysisClass::~analysisClass(): ends " << std::endl;
}

void analysisClass::Loop() {
  
  //--------------------------------------------------------------------------
  // Decide which plots to save (default is to save everything)
  //--------------------------------------------------------------------------
  
  fillSkim                         (  true  ) ;
  fillAllPreviousCuts              (  true  ) ;
  fillAllOtherCuts                 (  true  ) ;
  fillAllSameLevelAndLowerLevelCuts(  true  ) ;
  fillAllCuts                      (  true  ) ;
  
  //------------------------------------------------------------------
  // How many events to skim over?
  //------------------------------------------------------------------
  
  Long64_t nentries = fChain->GetEntries();
  std::cout << "analysisClass::Loop(): nentries = " << nentries << std::endl;   

  //------------------------------------------------------------------
  // Loop over events
  //------------------------------------------------------------------

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {

    //------------------------------------------------------------------
    // ROOT loop preamble
    //------------------------------------------------------------------

    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
    {
      std::cout << "ERROR: Could not read from TTree; exiting." << std::endl;
      exit(-1);
    }
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if (nb < 0)
    {
      std::cout << "ERROR: Could not read entry from TTree: read " << nb << "bytes; exiting." << std::endl;
      exit(-2);
    }

    //------------------------------------------------------------------
    // Tell user how many events we've looped over
    //------------------------------------------------------------------

    if(jentry < 10 || jentry%5000 == 0) std::cout << "analysisClass::Loop(): jentry = " << jentry << "/" << nentries << std::endl;   
    
    //-----------------------------------------------------------------
    // Get ready to fill variables 
    //-----------------------------------------------------------------

    resetCuts();

    //------------------------------------------------------------------
    // If this is data, it has to pass the trigger
    //------------------------------------------------------------------

    //// NB: trigger now is done later in the analysis class, so this information is ignored
    //float trigEff = 0.0;
    //int pass_trigger = 0;
    //if ( isData ) { 
    //  pass_trigger = 0;
    //  //if ( H_Ele27_WPLoose == 1)
    //  //if ( H_Ele27_WPTight == 1 || H_Photon175 == 1) // for 2016
    //  if ( H_Ele27_WPLoose_eta2p1 == 1)
    //    pass_trigger = 1;
    //}
    //else // using the turn-on in the MC
    //{
    //  // lead electron only
    //  if(Ele1_PtHeep > Ele2_PtHeep)
    //  {
    //    pass_trigger = trigEle27::passTrig(Ele1_PtHeep,Ele1_SCEta) ? 1 : 0;
    //    trigEff = trigEle27::turnOn(Ele1_PtHeep,Ele1_SCEta) ? 1 : 0;
    //  }
    //  else {
    //    pass_trigger = trigEle27::passTrig(Ele2_PtHeep,Ele2_SCEta) ? 1 : 0;
    //    trigEff = trigEle27::turnOn(Ele2_PtHeep,Ele2_SCEta) ? 1 : 0;
    //  }
    //}

    //------------------------------------------------------------------
    // Is the muon going to be Ele1 or Ele2?
    //------------------------------------------------------------------

    bool muon_is_Ele1 = false;
    bool muon_is_Ele2 = false;
    if   ( Muon1_Pt > 0.1 ) { 
      if   ( Muon1_Pt >  Ele1_Pt ) muon_is_Ele1 = true;
      else                         muon_is_Ele2 = true;
    }
    
    //------------------------------------------------------------------
    // If the muon is Ele2, then recalculate the Ele2 variables
    //------------------------------------------------------------------
    
    if ( muon_is_Ele2 ) {

      // Get object 4-vectors

      TLorentzVector muon, electron, jet1, jet2, met, e1e2, e2j1, e2j2;
      muon    .SetPtEtaPhiM ( Muon1_Pt, Muon1_Eta, Muon1_Phi, 0.0 );
      electron.SetPtEtaPhiM ( Ele1_Pt , Ele1_Eta , Ele1_Phi , 0.0 );
      jet1    .SetPtEtaPhiM ( Jet1_Pt , Jet1_Eta , Jet1_Phi , 0.0 );
      jet2    .SetPtEtaPhiM ( Jet2_Pt , Jet2_Eta , Jet2_Phi , 0.0 );
      met     .SetPtEtaPhiM ( PFMET_Type1XY_Pt, 0.0, PFMET_Type1XY_Phi, 0.0 );
      e1e2    = muon + electron;
      e2j1    = muon + jet1;
      e2j2    = muon + jet2;

      // These values can be re-calculated

      Ele2_Pt       = muon.Pt();
      Ele2_PtHeep   = muon.Pt();
      Ele2_Eta      = muon.Eta();
      Ele2_SCEta    = muon.Eta();
      Ele2_SCEt     = muon.Pt();
      Ele2_Phi      = muon.Phi();
      Ele2_Charge   = Muon1_Charge;

      DR_Ele2Jet1   = muon.DeltaR ( jet1 );
      DR_Ele2Jet2   = muon.DeltaR ( jet2 );
      mDPhi_METEle2 = fabs ( muon.DeltaPhi ( met ));
      
      M_e1e2        = e1e2.M();
      M_e2j1        = e2j1.M();
      M_e2j2        = e2j2.M();
      sT_eejj       = electron.Pt() + muon.Pt() + jet1.Pt() + jet2.Pt();
      Pt_e1e2       = e1e2.Pt();

      if ( nEle_store < 2) nEle_store = 2;
      // why do this? let's keep the original value instead, which could be 1 for emu events
      //if ( nEle_ptCut < 2) nEle_ptCut = 2;

      // These values cannot be re-calculated with the information stored

      Ele2_MissingHits    = -999.;
      Ele2_DCotTheta      = -999.;
      Ele2_Dist           = -999.;
      Ele2_Energy         = -999.;
      Ele2_SCEnergy       = -999.;
      Ele2_TrkEta         = -999.;
      Ele2_ValidFrac      = -999.;
      //Ele2_hltDoubleElePt = -999.;
      //FIXME in reduced skim
      //Ele2_hltEleSignalPt = -999.;
      //Ele1_hltEleTTbarPt  = trigEff;
      
    }

    else if ( muon_is_Ele1 ) { 
      
      // Get object 4-vectors

      TLorentzVector muon, electron, jet1, jet2, met, e1e2, e1j1, e1j2;
      muon    .SetPtEtaPhiM ( Muon1_Pt, Muon1_Eta, Muon1_Phi, 0.0 );
      electron.SetPtEtaPhiM ( Ele1_Pt , Ele1_Eta , Ele1_Phi , 0.0 );
      jet1    .SetPtEtaPhiM ( Jet1_Pt , Jet1_Eta , Jet1_Phi , 0.0 );
      jet2    .SetPtEtaPhiM ( Jet2_Pt , Jet2_Eta , Jet2_Phi , 0.0 );
      met     .SetPtEtaPhiM ( PFMET_Type1XY_Pt, 0.0, PFMET_Type1XY_Phi, 0.0 );
      e1e2    = muon + electron;
      e1j1    = muon + jet1;
      e1j2    = muon + jet2;

      // Demote the first electron (electron 1 -> electron 2)
      
      DR_Ele2Jet1          = DR_Ele1Jet1;         
      DR_Ele2Jet2	   = DR_Ele1Jet2;	   
      Ele2_Charge	   = Ele1_Charge;	   
      Ele2_DCotTheta	   = Ele1_DCotTheta;	   
      Ele2_Dist  	   = Ele1_Dist;	   
      Ele2_Energy	   = Ele1_Energy;	   
      Ele2_Eta		   = Ele1_Eta;		   
      Ele2_TrkEta    = Ele1_TrkEta;
      Ele2_MissingHits	   = Ele1_MissingHits;	   
      Ele2_Phi		   = Ele1_Phi;		   
      Ele2_Pt		   = Ele1_Pt;		   
      Ele2_PtHeep		   = Ele1_PtHeep;		   
      Ele2_SCEt    = Ele1_SCEt;
      Ele2_SCEta    = Ele1_SCEta;
      Ele2_SCPhi    = Ele1_SCPhi;
      //Ele2_hltDoubleElePt  = Ele1_hltDoubleElePt; 
      //FIXME in reduced skim
      //Ele2_hltEleSignalPt  = Ele1_hltEleSignalPt; 
      //Ele2_hltEleTTbarPt   = Ele1_hltEleTTbarPt;  
      mDPhi_METEle1	   = mDPhi_METEle1;	
      M_e2j1               = M_e1j1;
      M_e2j2               = M_e1j2;
      
      // These values can be re-calculated (muon 1 -> electron 1)

      Ele1_Pt       = muon.Pt();
      Ele1_PtHeep   = muon.Pt();
      Ele1_Eta      = muon.Eta();
      Ele1_SCEta    = muon.Eta();
      Ele1_SCEt     = muon.Pt();
      Ele1_Phi      = muon.Phi();
      Ele1_Charge   = Muon1_Charge;
      MT_Ele1MET    = sqrt ( 2.0 * muon.Pt() * met.Pt() * ( 1.0 - cos ( met.DeltaPhi(muon))));

      DR_Ele1Jet1   = muon.DeltaR ( jet1 );
      DR_Ele1Jet2   = muon.DeltaR ( jet2 );
      mDPhi_METEle1 = fabs ( muon.DeltaPhi ( met ));
      
      M_e1e2        = e1e2.M();
      M_e1j1        = e1j1.M();
      M_e1j2        = e1j2.M();
      sT_eejj       = electron.Pt() + muon.Pt() + jet1.Pt() + jet2.Pt();
      Pt_e1e2       = e1e2.Pt();

      if ( nEle_store < 2) nEle_store = 2;
      // why do this? let's keep the original value instead, which could be 1 for emu events
      //if ( nEle_ptCut < 2) nEle_ptCut = 2;
      
      // These values cannot be re-calculated with the information stored

      Ele1_MissingHits    = -999.;
      Ele1_DCotTheta      = -999.;
      Ele1_Dist           = -999.;
      Ele1_Energy         = -999.;
      Ele1_SCEnergy       = -999.;
      Ele1_TrkEta         = -999.;
      Ele1_ValidFrac      = -999.;
      //Ele1_hltDoubleElePt = -999.;
      //FIXME in reduced skim
      //Ele1_hltEleSignalPt = -999.;
      //Ele2_hltEleTTbarPt  = trigEff;
      
    }

    // std::cout << "N(muon, pt cut) = " << nMuon_ptCut << std::endl;

    Ele2_ValidFrac = 999.0; // this is a tag

    //------------------------------------------------------------------
    // Fill variables 
    //------------------------------------------------------------------
    
    fillVariableWithValue( "nEle_ptCut"  , nEle_ptCut   );
    fillVariableWithValue( "nMuon_ptCut" , nMuon_ptCut  );
    //fillVariableWithValue( "Muon1_Pt"    , Muon1_Pt     );
    //fillVariableWithValue( "Muon2_Pt"    , Muon2_Pt     );
    //fillVariableWithValue( "Ele1_Pt"     , Ele1_Pt      );	  
    //fillVariableWithValue( "Ele2_Pt"     , Ele2_Pt      );	  	  
    fillVariableWithValue( "Ele1_PtHeep"     , Ele1_PtHeep      );	  
    fillVariableWithValue( "Ele2_PtHeep"     , Ele2_PtHeep      );	  
    fillVariableWithValue( "Jet1_Pt"     , Jet1_Pt      );
    fillVariableWithValue( "Jet2_Pt"     , Jet2_Pt      );
    fillVariableWithValue( "sT_eejj"     , sT_eejj      );
    fillVariableWithValue( "M_e1e2"      , M_e1e2       );
    //fillVariableWithValue( "PassTrigger" , pass_trigger );
    fillVariableWithValue( "PassFilter"  , 1            );

    //-----------------------------------------------------------------
    // Evaluate the cuts
    //-----------------------------------------------------------------    

    evaluateCuts();
    
    //------------------------------------------------------------------
    // If event passes, fill the tree
    //------------------------------------------------------------------

    if ( passedCut            ("PassFilter") &&
	 passedAllPreviousCuts("PassFilter") ){
      fillSkimTree();
    }
  } // End loop over events
  
  std::cout << "analysisClass::Loop() ends" <<std::endl;   
}
