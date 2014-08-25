#include "Stdafx.h"
#include "GUI Simulation.h"
#include "clix.h"



GUISimulation::ManagedMultisliceSimulation::ManagedMultisliceSimulation()
{
	unmanagedMultislice = new UnmanagedMultisliceSimulation();
}

GUISimulation::ManagedMultisliceSimulation::~ManagedMultisliceSimulation()
{
	delete unmanagedMultislice;
}

void GUISimulation::ManagedMultisliceSimulation::ImportAtoms(String^ filename)
{
	using namespace clix;

	std::string cfilename = marshalString<E_ANSI>(filename);
	unmanagedMultislice->LoadAtomFile(cfilename);
}

void GUISimulation::ManagedMultisliceSimulation::UploadBinnedAtoms()
{
	unmanagedMultislice->UploadBinnedAtoms();
}

void GUISimulation::ManagedMultisliceSimulation::UploadBinnedAtomsTDS()
{
	unmanagedMultislice->UploadBinnedAtomsTDS();
}

void GUISimulation::ManagedMultisliceSimulation::GetParameterisation()
{
	unmanagedMultislice->GetParameterisation();
}

void GUISimulation::ManagedMultisliceSimulation::ApplyMicroscopeParameters(float Voltage, float defocus, float Mod2f, float Arg2f, float spherical, float B, float D, float obj)
{
	unmanagedMultislice->ApplyMicroscopeParameters(Voltage, defocus, Mod2f, Arg2f, spherical, B, D, obj);
}

void GUISimulation::ManagedMultisliceSimulation::SetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float SizeX, float SizeY, float SizeZ, float blockoffsetx, float blockoffsety)
{
	unmanagedMultislice->SetCalculationVariables(PixelScaleIn, defocus, resX, resY, SizeX, SizeY, SizeZ, blockoffsetx,blockoffsety);
}

void GUISimulation::ManagedMultisliceSimulation::STEMSetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float SizeX, float SizeY, float SizeZ)
{
	unmanagedMultislice->STEMSetCalculationVariables(PixelScaleIn, defocus, resX, resY, SizeX, SizeY, SizeZ);
}

void GUISimulation::ManagedMultisliceSimulation::PreCalculateFrequencies()
{
	unmanagedMultislice->PreCalculateFrequencies();
}

void GUISimulation::ManagedMultisliceSimulation::InitialiseWavefunctions()
{
	unmanagedMultislice->InitialiseWavefunctions();
}

void GUISimulation::ManagedMultisliceSimulation::InitialiseWavefunctions(float dz)
{
	unmanagedMultislice->InitialiseWavefunctions(dz);

}
void GUISimulation::ManagedMultisliceSimulation::STEMInitialiseWavefunctions(int posx, int posy)
{
	unmanagedMultislice->STEMInitialiseWavefunctions(posx,posy);
}
void GUISimulation::ManagedMultisliceSimulation::MultisliceStepFD(int iteration)
{
	unmanagedMultislice->MultisliceStep(iteration);
}

void GUISimulation::ManagedMultisliceSimulation::AllocTDS()
{
	unmanagedMultislice->AllocTDS();

}
void GUISimulation::ManagedMultisliceSimulation::MultisliceStepConventional(int iteration)
{
	unmanagedMultislice->MultisliceStepConv(iteration);
}

float GUISimulation::ManagedMultisliceSimulation::GetSizeX()
{
	return (unmanagedMultislice->MaximumX-unmanagedMultislice->MinimumX);
}

float GUISimulation::ManagedMultisliceSimulation::GetSizeY()
{
	return (unmanagedMultislice->MaximumY-unmanagedMultislice->MinimumY);
}

float GUISimulation::ManagedMultisliceSimulation::GetSizeZ()
{
	return (unmanagedMultislice->MaximumZ-unmanagedMultislice->MinimumZ);
}

//TODO: change all dz's for atom slice so they aren't independent
int GUISimulation::ManagedMultisliceSimulation::GetSlices(bool conventional)
{
	if(conventional)
		return (ceil((unmanagedMultislice->MaximumZ - unmanagedMultislice->MinimumZ) / unmanagedMultislice->conventionaldz ));
	else
		return (unmanagedMultislice->numberOfSlices);
}

float GUISimulation::ManagedMultisliceSimulation::GetKmax()
{
	return (unmanagedMultislice->kmax);
}

void GUISimulation::ManagedMultisliceSimulation::GetExitWave()
{
	unmanagedMultislice->GetExitWave();
}

void GUISimulation::ManagedMultisliceSimulation::GetSTEMExitWave()
{
	unmanagedMultislice->GetSTEMExitWave();
}

void GUISimulation::ManagedMultisliceSimulation::AddTDSWaves()
{
	unmanagedMultislice->AddTDSWaves();
}


void GUISimulation::ManagedMultisliceSimulation::FreeExitWave()
{
	unmanagedMultislice->FreeExitWave();
}

float GUISimulation::ManagedMultisliceSimulation::GetEWValueAbs(int xpos, int ypos)
{
	return(unmanagedMultislice->GetEWValueAbs(xpos,ypos));
}


float GUISimulation::ManagedMultisliceSimulation::GetEWValueRe(int xpos, int ypos)
{
	return(unmanagedMultislice->GetEWValueRe(xpos,ypos));
}

float GUISimulation::ManagedMultisliceSimulation::GetEWValueIm(int xpos, int ypos)
{
	return(unmanagedMultislice->GetEWValueIm(xpos,ypos));
}

float GUISimulation::ManagedMultisliceSimulation::GetImValue(int xpos, int ypos)
{
	return(unmanagedMultislice->GetImValue(xpos,ypos));
}

float GUISimulation::ManagedMultisliceSimulation::GetTDSMin()
{
	return(unmanagedMultislice->GetTDSMin());
}

float GUISimulation::ManagedMultisliceSimulation::GetTDSMax()
{
	return(unmanagedMultislice->GetTDSMax());
}

float GUISimulation::ManagedMultisliceSimulation::GetTDSVal(int xpos, int ypos)
{
	return(unmanagedMultislice->GetTDSVal(xpos,ypos));
}

void GUISimulation::ManagedMultisliceSimulation::SimulateImage(float doseperpix)
{
	unmanagedMultislice->SimulateImage(doseperpix);
}

void GUISimulation::ManagedMultisliceSimulation::AllocHostImage()
{
	unmanagedMultislice->AllocHostImage();
}

float GUISimulation::ManagedMultisliceSimulation::GetImMax(){

		return(unmanagedMultislice->maxposIm);
}

float GUISimulation::ManagedMultisliceSimulation::GetEWMax(){

		return(unmanagedMultislice->maxposEW);
}

float GUISimulation::ManagedMultisliceSimulation::GetEWMin(){

		return(unmanagedMultislice->minposEW);
}

float GUISimulation::ManagedMultisliceSimulation::GetImMin(){

		return(unmanagedMultislice->minposIm);
}

float GUISimulation::ManagedMultisliceSimulation::GetImageContrast(int t, int l, int b, int r){

	// need to work on setting t,l,b,r first.

		return(unmanagedMultislice->imagecontrast);
}
