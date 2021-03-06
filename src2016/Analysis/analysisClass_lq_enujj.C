#define analysisClass_cxx
#include "analysisClass.h"
#include <TH2.h>
#include <TH1F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <TVector2.h>
#include <TVector3.h>

analysisClass::analysisClass(string * inputList, string * cutFile, string * treeName, string * outputFileName, string * cutEfficFile)
  :baseClass(inputList, cutFile, treeName, outputFileName, cutEfficFile){}

analysisClass::~analysisClass(){}

void analysisClass::Loop()
{
   std::cout << "analysisClass::Loop() begins" <<std::endl;   

   //--------------------------------------------------------------------------
   // Decide which plots to save (default is to save everything)
   //--------------------------------------------------------------------------
   
   fillSkim                         (  !true  ) ;
   fillAllPreviousCuts              ( !true  ) ;
   fillAllOtherCuts                 (  true  ) ;
   fillAllSameLevelAndLowerLevelCuts( !true  ) ;
   fillAllCuts                      ( !true  ) ;
   
   //--------------------------------------------------------------------------
   // Create TH1D's
   //--------------------------------------------------------------------------

   CreateUserTH1D( "ProcessID"             ,    21  , -0.5    , 20.5     );
   CreateUserTH1D( "ProcessID_PAS"         ,    21  , -0.5    , 20.5     );
   CreateUserTH1D( "ProcessID_WWindow"     ,    21  , -0.5    , 20.5     );
   
   CreateUserTH1D( "nElectron_PAS"            , 5   , -0.5    , 4.5      );
   CreateUserTH1D( "nMuon_PAS"                , 5   , -0.5    , 4.5      );
   CreateUserTH1D( "nJet_PAS"                 , 11  , -0.5    , 10.5     );
   CreateUserTH1D( "Pt1stEle_PAS"	      , 100 , 0       , 1000     ); 
   CreateUserTH1D( "Eta1stEle_PAS"	      , 100 , -5      , 5	 ); 
   CreateUserTH1D( "Phi1stEle_PAS"	      , 60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "Charge1stEle_PAS"	      , 2   , -1.0001 , 1.0001	 ); 
   CreateUserTH1D( "MET_PAS"                  , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "METPhi_PAS"		      , 60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "MET_Type01_PAS"           , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "MET_Type01_Phi_PAS"	      , 60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "minMETPt1stEle_PAS"       , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "minMET01Pt1stEle_PAS"     , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "Pt1stJet_PAS"             , 100 , 0       , 1000	 ); 
   CreateUserTH1D( "Pt2ndJet_PAS"             , 100 , 0       , 1000	 ); 
   CreateUserTH1D( "Eta1stJet_PAS"            , 100 , -5      , 5	 ); 
   CreateUserTH1D( "Eta2ndJet_PAS"            , 100 , -5      , 5	 ); 
   CreateUserTH1D( "Phi1stJet_PAS"	      , 60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "Phi2ndJet_PAS"	      , 60  , -3.1416 , +3.1416	 ); 
   CreateUserTH1D( "CSV1stJet_PAS"            , 200 , 0       , 1.0	 ); 
   CreateUserTH1D( "CSV2ndJet_PAS"            , 200 , 0       , 1.0	 ); 
   CreateUserTH1D( "nMuon_PtCut_IDISO_PAS"    , 16  , -0.5    , 15.5	 ); 
   CreateUserTH1D( "MTenu_PAS"                , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "MTenu_Type01_PAS"         , 200 , 0       , 1000	 ); 
   CreateUserTH1D( "Ptenu_PAS"		      , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "sTlep_PAS"                , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "sTlep_Type01_PAS"         , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "sTjet_PAS"                , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "sT_PAS"                   , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "sT_Type01_PAS"            , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "Mjj_PAS"		      , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "Mej1_PAS"                 , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "Mej2_PAS"                 , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "Mej_PAS"                  , 200 , 0       , 2000	 ); 
   CreateUserTH1D( "MTjnu_PAS"                , 200 , 0       , 1000     );
   CreateUserTH1D( "DCotTheta1stEle_PAS"      , 100 , 0.0     , 1.0      );
   CreateUserTH1D( "Dist1stEle_PAS"           , 100 , 0.0     , 1.0      );  
   CreateUserTH1D( "DR_Ele1Jet1_PAS"	      , 100 , 0       , 10       ); 
   CreateUserTH1D( "DR_Ele1Jet2_PAS"	      , 100 , 0       , 10       ); 
   CreateUserTH1D( "DR_Jet1Jet2_PAS"	      , 100 , 0       , 10       );
   CreateUserTH1D( "minDR_EleJet_PAS"         , 100 , 0       , 10       );  
   CreateUserTH1D( "mDPhi1stEleMET_PAS"       , 100 , 0.      , 3.14159  );
   CreateUserTH1D( "mDPhi1stJetMET_PAS"       , 100 , 0.      , 3.14159  );
   CreateUserTH1D( "mDPhi2ndJetMET_PAS"       , 100 , 0.      , 3.14159  );
   CreateUserTH1D( "mDPhi1stEleMET_Type01_PAS", 100 , 0.      , 3.14159  );
   CreateUserTH1D( "mDPhi1stJetMET_Type01_PAS", 100 , 0.      , 3.14159  );
   CreateUserTH1D( "mDPhi2ndJetMET_Type01_PAS", 100 , 0.      , 3.14159  );
   CreateUserTH1D( "GeneratorWeight"          , 200 , -2.0    , 2.0      );
   CreateUserTH1D( "PileupWeight"             , 200 , -2.0    , 2.0      );

   CreateUserTH1D( "Pt1stEle_Pt40to45_EtaGT2p1", 150, 35., 50. );

   CreateUserTH1D( "MT_GoodVtxLTE3_PAS"       , 200 , 0.      ,  1000    );
   CreateUserTH1D( "MT_GoodVtxGTE4_LTE8_PAS"  , 200 , 0.      ,  1000    );
   CreateUserTH1D( "MT_GoodVtxGTE9_LTE15_PAS" , 200 , 0.      ,  1000    );
   CreateUserTH1D( "MT_GoodVtxGTE16_PAS"      , 200 , 0.      ,  1000    );

   CreateUserTH1D( "nVertex_PAS"           ,    101   , -0.5   , 100.5	 ) ; 
   
   CreateUserTH1D( "MTenu_50_110", 200, 40, 140 );
   CreateUserTH1D( "nJets_MTenu_50_110"    , 20 , -0.5, 19.5 );
   CreateUserTH1D( "MTenu_50_110_Njet_gte4", 200, 40, 140 );
   CreateUserTH1D( "MTenu_50_110_Njet_lte3", 200, 40, 140 );
   CreateUserTH1D( "MTenu_50_110_Njet_lte4", 200, 40, 140 );
   CreateUserTH1D( "MTenu_50_110_Njet_gte5", 200, 40, 140 );

   CreateUserTH1D( "MTenu_Type01_50_110", 200, 40, 140 );
   CreateUserTH1D( "nJets_MTenu_Type01_50_110"    , 20 , -0.5, 19.5 );
   CreateUserTH1D( "MTenu_Type01_50_110_Njet_gte4", 200, 40, 140 );
   CreateUserTH1D( "MTenu_Type01_50_110_Njet_lte3", 200, 40, 140 );
   CreateUserTH1D( "MTenu_Type01_50_110_Njet_lte4", 200, 40, 140 );
   CreateUserTH1D( "MTenu_Type01_50_110_Njet_gte5", 200, 40, 140 );

   CreateUserTH1D( "run_PAS"               ,    20000 , 160300  , 180300 );
   CreateUserTH1D( "run_HLT"               ,    20000 , 160300  , 180300 );
   
   //--------------------------------------------------------------------------
   // Loop over the chain
   //--------------------------------------------------------------------------

   if (fChain == 0) return;
   std::cout << "fChain = " << fChain << std::endl;

   Long64_t nentries = fChain->GetEntries();
   std::cout << "analysisClass::Loop(): nentries = " << nentries << std::endl;   

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     Long64_t ientry = LoadTree(jentry);
     if (ientry < 0) break;
     nb = fChain->GetEntry(jentry);   nbytes += nb;
     if(jentry < 10 || jentry%1000 == 0) std::cout << "analysisClass::Loop(): jentry = " << jentry << "/" << nentries << std::endl;   

     //--------------------------------------------------------------------------
     // Reset the cuts
     //--------------------------------------------------------------------------

     resetCuts();

     //--------------------------------------------------------------------------
     // Do pileup re-weighting
     //--------------------------------------------------------------------------
     
     double pileup_weight = getPileupWeight ( nPileUpInt_True, isData ) ;
     
     //--------------------------------------------------------------------------
     // Get information about gen-level reweighting (should be for Sherpa only)
     //--------------------------------------------------------------------------

     double gen_weight = Weight;
     if ( isData ) gen_weight = 1.0;

     //--------------------------------------------------------------------------
     // Check good run list
     //--------------------------------------------------------------------------
     
     int passedJSON = passJSON ( run, ls , isData ) ;
     if ( !isData ) passedJSON = 1;
     
     //--------------------------------------------------------------------------
     // Check HLT
     //--------------------------------------------------------------------------
     
     // Fill HLT 
     int passHLT = 1;
     if ( isData ) { 
       passHLT = 0;
       if ( H_Ele30_PFJet100_25      == 1 ||
	    H_Ele30_PFNoPUJet100_25  == 1 ){
       	 passHLT = 1;
       }
     }

     //--------------------------------------------------------------------------
     // Fill variables
     //--------------------------------------------------------------------------

     // JSON variable
     fillVariableWithValue(   "Reweighting"              , 1                       , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassJSON"                 , passedJSON              , gen_weight * pileup_weight ); 
     				
     // HLT variable							           
     fillVariableWithValue(   "PassHLT"                  , passHLT                 , gen_weight * pileup_weight );
     
     // Noise filters
     fillVariableWithValue(   "PassHBHENoiseFilter"	      , PassHBHENoiseFilter                              , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassBeamHaloFilterTight"       , PassBeamHaloFilterTight                          , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassBadEESupercrystalFilter"   , ( isData == 1 ) ? PassBadEESupercrystalFilter : 1, gen_weight * pileup_weight );
     fillVariableWithValue(   "PassBeamScraping"	      , ( isData == 1 ) ? PassBeamScraping	      : 1, gen_weight * pileup_weight );
     fillVariableWithValue(   "PassEcalDeadCellBoundEnergy"   , PassEcalDeadCellBoundEnergy                      , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassEcalDeadCellTrigPrim"      , PassEcalDeadCellTrigPrim                         , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassEcalLaserCorrFilter"       , ( isData == 1 ) ? PassEcalLaserCorrFilter     : 1, gen_weight * pileup_weight );
     fillVariableWithValue(   "PassHcalLaserEventFilter"      , ( isData == 1 ) ? PassHcalLaserEventFilter    : 1, gen_weight * pileup_weight );
     fillVariableWithValue(   "PassPhysDecl"		      , ( isData == 1 ) ? PassPhysDecl		      : 1, gen_weight * pileup_weight );
     fillVariableWithValue(   "PassPrimaryVertex"	      , PassPrimaryVertex                                , gen_weight * pileup_weight );
     fillVariableWithValue(   "PassTrackingFailure"	      , ( isData == 1 ) ? PassTrackingFailure	      : 1, gen_weight * pileup_weight );
     
     // Muon variables ( for veto ) 					      
     fillVariableWithValue(   "nMuon"                    , nMuon_ptCut             , gen_weight * pileup_weight );
			                                      		                
     // 1st Electron variables				      		                
     fillVariableWithValue(   "nEle"                     , nEle_ptCut              , gen_weight * pileup_weight ); 
     fillVariableWithValue(   "Ele1_Pt"                  , Ele1_Pt                 , gen_weight * pileup_weight );
     fillVariableWithValue(   "MTenu"                    , MT_Ele1MET              , gen_weight * pileup_weight );

     // MET variables	                                      		           
     fillVariableWithValue(   "MET"                      , PFMET_Type01XY_Pt       , gen_weight * pileup_weight );
     fillVariableWithValue(   "mDeltaPhiMETEle"          , mDPhi_METEle1           , gen_weight * pileup_weight );
     									           
     // 1st JET variables                                     		           
     fillVariableWithValue(   "nJet"                     , nJet_ptCut              , gen_weight * pileup_weight );
				
     double MT_Jet1MET, MT_Jet2MET, MT_Ele1Jet1, MT_Ele1Jet2, MT_Ele1MET_Type01;
     double mDPhi_METType01_Ele1, mDPhi_METType01_Jet1, mDPhi_METType01_Jet2;

     // alternate METs

     if ( nEle_store > 0 ) {
       TVector2 v_ele;
       TVector2 v_MET_Type01;
       v_MET_Type01.SetMagPhi( PFMET_Type01_Pt , PFMET_Type01_Phi  );
       v_ele.SetMagPhi( Ele1_Pt, Ele1_Phi );
       mDPhi_METType01_Ele1 = fabs(v_MET_Type01.DeltaPhi ( v_ele ));
       float deltaphi = v_MET_Type01.DeltaPhi(v_ele);
       MT_Ele1MET_Type01 = sqrt ( 2 * Ele1_Pt * PFMET_Type01_Pt * ( 1 - cos ( deltaphi ) ) );
     }
     
     // 1st JET variables                                     		           
     if ( nJet_store > 0 ) { 						           
       fillVariableWithValue( "Jet1_Pt"                  , Jet1_Pt                 , gen_weight * pileup_weight );
       fillVariableWithValue( "Jet1_Eta"                 , Jet1_Eta                , gen_weight * pileup_weight );
       fillVariableWithValue( "mDeltaPhiMET1stJet"       , mDPhi_METJet1           , gen_weight * pileup_weight );

       TVector2 v_MET;
       TVector2 v_jet;
       TVector2 v_MET_Type01;
       v_MET_Type01.SetMagPhi( PFMET_Type01_Pt , PFMET_Type01_Phi  );
       v_MET.SetMagPhi( PFMET_Type01XY_Pt , PFMET_Type01XY_Phi  );
       v_jet.SetMagPhi( Jet1_Pt, Jet1_Phi );
       mDPhi_METType01_Jet1 = fabs(v_MET_Type01.DeltaPhi ( v_jet ));
       float deltaphi = v_MET.DeltaPhi(v_jet);
       MT_Jet1MET = sqrt ( 2 * Jet1_Pt * PFMET_Type01XY_Pt * ( 1 - cos ( deltaphi ) ) );
     }									           
     
     // 2nd JET variables                                     		           
     if ( nJet_store > 1 ) { 	                                      	           
       fillVariableWithValue( "Jet2_Pt"                  , Jet2_Pt                 , gen_weight * pileup_weight );
       fillVariableWithValue( "Jet2_Eta"                 , Jet2_Eta                , gen_weight * pileup_weight );
       fillVariableWithValue( "ST"                       , sT_enujj                , gen_weight * pileup_weight );
       
       TVector2 v_MET;
       TVector2 v_jet;
       TVector2 v_MET_Type01;
       v_MET_Type01.SetMagPhi( PFMET_Type01_Pt , PFMET_Type01_Phi  );
       v_MET.SetMagPhi( PFMET_Type01XY_Pt , PFMET_Type01XY_Phi );
       v_jet.SetMagPhi( Jet2_Pt, Jet2_Phi );
       mDPhi_METType01_Jet2 = fabs(v_MET_Type01.DeltaPhi ( v_jet ));
       float deltaphi = v_MET.DeltaPhi(v_jet);
       MT_Jet2MET = sqrt ( 2 * Jet2_Pt * PFMET_Type01XY_Pt * ( 1 - cos ( deltaphi ) ) );
     }
     
     // 3rd JET variables 
     // if ( nJet_store > 2 ) {
     //   fillVariableWithValue( "Jet3_Pt"                  , Jet3_Pt                 , gen_weight * pileup_weight );
     //   fillVariableWithValue( "Jet3_Eta"                 , Jet3_Eta                , gen_weight * pileup_weight );
     // }

     // 1 electron, 1 jet variables 
     if ( nEle_ptCut > 0 && nJet_ptCut > 0 ) { 
       fillVariableWithValue ( "DR_Ele1Jet1"             , DR_Ele1Jet1             , gen_weight * pileup_weight );

       TVector2 v_ele;
       TVector2 v_jet1;
       v_ele .SetMagPhi ( Ele1_Pt, Ele1_Phi );
       v_jet1.SetMagPhi ( Jet1_Pt, Jet1_Phi );
       float deltaphi = v_ele.DeltaPhi ( v_jet1 );
       MT_Ele1Jet1 = sqrt ( 2 * Jet1_Pt * Ele1_Pt * ( 1 - cos ( deltaphi ) ) );
     }

     // 1 electron, 2 jet variables 
     if ( nEle_store > 0 && nJet_store > 1 ) { 
       fillVariableWithValue ( "DR_Ele1Jet2"             , DR_Ele1Jet2             , gen_weight * pileup_weight );

       TVector2 v_ele;
       TVector2 v_jet2;
       v_ele .SetMagPhi ( Ele1_Pt, Ele1_Phi );
       v_jet2.SetMagPhi ( Jet2_Pt, Jet2_Phi );
       float deltaphi = v_ele.DeltaPhi ( v_jet2 );
       MT_Ele1Jet2 = sqrt ( 2 * Jet2_Pt * Ele1_Pt * ( 1 - cos ( deltaphi ) ) );
     }

     double MT_JetMET;
     double MT_EleJet;
     double Mej;
     
     if ( fabs ( MT_Jet1MET - MT_Ele1Jet2 ) < fabs( MT_Jet2MET - MT_Ele1Jet1 )){
       MT_JetMET = MT_Jet1MET;
       MT_EleJet = MT_Ele1Jet2;
       Mej = M_e1j2;
     } else { 
       MT_JetMET = MT_Jet2MET;
       MT_EleJet = MT_Ele1Jet1;
       Mej = M_e1j1;
     }	 

     // Dummy variables
     fillVariableWithValue ("preselection",1, gen_weight * pileup_weight ); 

     //--------------------------------------------------------------------------
     // Evaluate the cuts
     //--------------------------------------------------------------------------
     
     evaluateCuts();

     if (!isData && !passedCut ("PassJSON")){
       std::cout << "ERROR: This event did not pass the JSON file!" << std::endl;
       std::cout << "  isData = " << isData << std::endl;
       std::cout << "  passedJSON = " << passedJSON << std::endl;
     }
     
     //--------------------------------------------------------------------------
     // Fill preselection plots
     //--------------------------------------------------------------------------
     
     bool passed_preselection = passedAllPreviousCuts("preselection");
     bool passed_minimum      = ( passedAllPreviousCuts("PassTrackingFailure") && passedCut ("PassTrackingFailure"));
    
     if ( passed_minimum && isData ){ 
       FillUserTH1D ("run_HLT", run );
     }

     FillUserTH1D( "ProcessID"      , ProcessID, pileup_weight * gen_weight ) ;
     
     if ( passed_preselection ) { 

       FillUserTH1D( "ProcessID_PAS"      , ProcessID, pileup_weight * gen_weight ) ;
       
       //--------------------------------------------------------------------------
       // Fill skim tree, if necessary
       //--------------------------------------------------------------------------

       double min_DR_EleJet = 999.0;
       double DR_Ele1Jet3 = 999.0;
       if ( nJet_store > 2 ) {
	 TLorentzVector ele1,  jet3;
	 ele1.SetPtEtaPhiM ( Ele1_Pt, Ele1_Eta, Ele1_Phi, 0.0 );
	 jet3.SetPtEtaPhiM ( Jet3_Pt, Jet3_Eta, Jet3_Phi, 0.0 );	 
	 DR_Ele1Jet3 = ele1.DeltaR ( jet3 ) ;
       }

       if ( DR_Ele1Jet1 < min_DR_EleJet ) min_DR_EleJet = DR_Ele1Jet1;
       if ( DR_Ele1Jet2 < min_DR_EleJet ) min_DR_EleJet = DR_Ele1Jet2;
       if ( nJet_store > 2 ) {
	 if ( DR_Ele1Jet3 < min_DR_EleJet ) min_DR_EleJet = DR_Ele1Jet3;
       }
              
       if ( isData )        FillUserTH1D("run_PAS", run ) ;
       FillUserTH1D( "nElectron_PAS"              , nEle_ptCut                                   , pileup_weight * gen_weight); 
       FillUserTH1D( "nMuon_PAS"                  , nMuon_ptCut                                  , pileup_weight * gen_weight); 
       FillUserTH1D( "Pt1stEle_PAS"	          , Ele1_Pt                                      , pileup_weight * gen_weight); 
       FillUserTH1D( "Eta1stEle_PAS"	          , Ele1_Eta                                     , pileup_weight * gen_weight);
       FillUserTH1D( "Phi1stEle_PAS"	          , Ele1_Phi                                     , pileup_weight * gen_weight);
       FillUserTH1D( "Charge1stEle_PAS"           , Ele1_Charge                                  , pileup_weight * gen_weight);   
       FillUserTH1D( "MET_PAS"                    , PFMET_Type01XY_Pt                            , pileup_weight * gen_weight);
       FillUserTH1D( "METPhi_PAS"	          , PFMET_Type01XY_Phi                           , pileup_weight * gen_weight);   
       FillUserTH1D( "MET_Type01_PAS"             , PFMET_Type01_Pt                              , pileup_weight * gen_weight);
       FillUserTH1D( "MET_Type01_Phi_PAS"	  , PFMET_Type01_Phi                             , pileup_weight * gen_weight);   
       FillUserTH1D( "minMETPt1stEle_PAS"         , TMath::Min ( Ele1_Pt, PFMET_Type01XY_Pt  )   , pileup_weight * gen_weight);
       FillUserTH1D( "minMET01Pt1stEle_PAS"       , TMath::Min ( Ele1_Pt, PFMET_Type01_Pt    )   , pileup_weight * gen_weight);
       FillUserTH1D( "Pt1stJet_PAS"               , Jet1_Pt                                      , pileup_weight * gen_weight);
       FillUserTH1D( "Pt2ndJet_PAS"               , Jet2_Pt                                      , pileup_weight * gen_weight);
       FillUserTH1D( "Eta1stJet_PAS"              , Jet1_Eta                                     , pileup_weight * gen_weight);
       FillUserTH1D( "Eta2ndJet_PAS"              , Jet2_Eta                                     , pileup_weight * gen_weight);
       FillUserTH1D( "Phi1stJet_PAS"              , Jet1_Phi                                     , pileup_weight * gen_weight);
       FillUserTH1D( "Phi2ndJet_PAS"	          , Jet2_Phi                                     , pileup_weight * gen_weight);
       FillUserTH1D( "CSV1stJet_PAS"              , Jet1_btagCSV                                 , pileup_weight * gen_weight);
       FillUserTH1D( "CSV2ndJet_PAS"              , Jet2_btagCSV                                 , pileup_weight * gen_weight);
       FillUserTH1D( "nMuon_PtCut_IDISO_PAS"      , nMuon_ptCut                                  , pileup_weight * gen_weight); 
       FillUserTH1D( "MTenu_PAS"                  , MT_Ele1MET                                   , pileup_weight * gen_weight);
       FillUserTH1D( "MTenu_Type01_PAS"           , MT_Ele1MET_Type01                            , pileup_weight * gen_weight);
       FillUserTH1D( "Ptenu_PAS"	          , Pt_Ele1MET                                   , pileup_weight * gen_weight);
       FillUserTH1D( "sTlep_PAS"                  , Ele1_Pt + PFMET_Type01XY_Pt                  , pileup_weight * gen_weight);
       FillUserTH1D( "sTlep_Type01_PAS"           , Ele1_Pt + PFMET_Type01_Pt                    , pileup_weight * gen_weight);
       FillUserTH1D( "sT_PAS"                     , sT_enujj                                     , pileup_weight * gen_weight);
       FillUserTH1D( "sT_Type01_PAS"              , Ele1_Pt + PFMET_Type01_Pt + Jet1_Pt + Jet2_Pt, pileup_weight * gen_weight);
       FillUserTH1D( "sTjet_PAS"                  , Jet1_Pt + Jet2_Pt                            , pileup_weight * gen_weight);
       FillUserTH1D( "Mjj_PAS"	                  , M_j1j2                                       , pileup_weight * gen_weight);   
       FillUserTH1D( "DCotTheta1stEle_PAS"        , Ele1_DCotTheta                               , pileup_weight * gen_weight);
       FillUserTH1D( "Dist1stEle_PAS"             , Ele1_Dist                                    , pileup_weight * gen_weight);
       FillUserTH1D( "mDPhi1stEleMET_PAS"         , mDPhi_METEle1                                , pileup_weight * gen_weight);
       FillUserTH1D( "mDPhi1stJetMET_PAS"         , mDPhi_METJet1                                , pileup_weight * gen_weight);
       FillUserTH1D( "mDPhi2ndJetMET_PAS"         , mDPhi_METJet2                                , pileup_weight * gen_weight); 
       FillUserTH1D( "mDPhi1stEleMET_Type01_PAS"  , mDPhi_METType01_Ele1                         , pileup_weight * gen_weight);
       FillUserTH1D( "mDPhi1stJetMET_Type01_PAS"  , mDPhi_METType01_Jet1                         , pileup_weight * gen_weight);
       FillUserTH1D( "mDPhi2ndJetMET_Type01_PAS"  , mDPhi_METType01_Jet2                         , pileup_weight * gen_weight); 
       FillUserTH1D( "Mej1_PAS"                   , M_e1j1                                       , pileup_weight * gen_weight);
       FillUserTH1D( "Mej2_PAS"                   , M_e1j2                                       , pileup_weight * gen_weight);
       FillUserTH1D( "Mej_PAS"                    , Mej                                          , pileup_weight * gen_weight);
       FillUserTH1D( "MTjnu_PAS"                  , MT_JetMET                                    , pileup_weight * gen_weight);
       FillUserTH1D( "DR_Ele1Jet1_PAS"	          , DR_Ele1Jet1                                  , pileup_weight * gen_weight);
       FillUserTH1D( "DR_Ele1Jet2_PAS"	          , DR_Ele1Jet2                                  , pileup_weight * gen_weight);
       FillUserTH1D( "DR_Jet1Jet2_PAS"	          , DR_Jet1Jet2                                  , pileup_weight * gen_weight);
       FillUserTH1D( "minDR_EleJet_PAS"           , min_DR_EleJet                                , pileup_weight * gen_weight);
       FillUserTH1D( "nVertex_PAS"                , nVertex                                      , pileup_weight * gen_weight);
       FillUserTH1D( "nJet_PAS"                   , nJet_ptCut                                   , pileup_weight * gen_weight);
       FillUserTH1D( "GeneratorWeight"            , gen_weight                                                               );
       FillUserTH1D( "PileupWeight"               , pileup_weight                                                            );


       if ( Ele1_Pt > 40 && Ele1_Pt < 45 && fabs ( Ele1_Eta ) > 2.1 ){
	 FillUserTH1D( "Pt1stEle_Pt40to45_EtaGT2p1", Ele1_Pt, pileup_weight * gen_weight);
       }
       
       if ( MT_Ele1MET > 50 && MT_Ele1MET < 110 ){
	 
	 FillUserTH1D( "ProcessID_WWindow", ProcessID, pileup_weight * gen_weight );
	 FillUserTH1D( "MTenu_50_110"      , MT_Ele1MET,  pileup_weight * gen_weight ) ;
	 FillUserTH1D( "nJets_MTenu_50_110", nJet_ptCut,  pileup_weight * gen_weight ) ;

	 if ( nJet_ptCut <= 3 ){ 
	   FillUserTH1D(   "MTenu_50_110_Njet_lte3", MT_Ele1MET,  pileup_weight * gen_weight ) ;
	 }

	 if ( nJet_ptCut <= 4 ){ 
	   FillUserTH1D(   "MTenu_50_110_Njet_lte4", MT_Ele1MET,  pileup_weight * gen_weight ) ;
	 }

	 if ( nJet_ptCut >= 4 ){ 
	   FillUserTH1D(   "MTenu_50_110_Njet_gte4", MT_Ele1MET,  pileup_weight * gen_weight ) ;
	 }
	 
	 if ( nJet_ptCut >= 5 ){ 
	   FillUserTH1D(   "MTenu_50_110_Njet_gte5", MT_Ele1MET,  pileup_weight * gen_weight ) ;
	 }
       }
       
       if ( MT_Ele1MET_Type01 > 50 && MT_Ele1MET_Type01 < 110 ){
	 
	 FillUserTH1D( "MTenu_Type01_50_110"      , MT_Ele1MET_Type01,  pileup_weight * gen_weight ) ;
	 FillUserTH1D( "nJets_MTenu_Type01_50_110", nJet_ptCut,  pileup_weight * gen_weight ) ;

	 if ( nJet_ptCut <= 3 ){ 
	   FillUserTH1D(   "MTenu_Type01_50_110_Njet_lte3", MT_Ele1MET_Type01,  pileup_weight * gen_weight ) ;
	 }

	 if ( nJet_ptCut <= 4 ){ 
	   FillUserTH1D(   "MTenu_Type01_50_110_Njet_lte4", MT_Ele1MET_Type01,  pileup_weight * gen_weight ) ;
	 }

	 if ( nJet_ptCut >= 4 ){ 
	   FillUserTH1D(   "MTenu_Type01_50_110_Njet_gte4", MT_Ele1MET_Type01,  pileup_weight * gen_weight ) ;
	 }
	 
	 if ( nJet_ptCut >= 5 ){ 
	   FillUserTH1D(   "MTenu_Type01_50_110_Njet_gte5", MT_Ele1MET_Type01,  pileup_weight * gen_weight ) ;
	 }
       }
     }
   } // End loop over events

   std::cout << "analysisClass::Loop() ends" <<std::endl;   
}
