// GUI Simulation.h

#include "unmanagedGUISimulation.cuh"

#pragma once

using namespace System;

#pragma managed
namespace GUISimulation {

	public ref class ManagedMultisliceSimulation
	{
	private:
		UnmanagedMultisliceSimulation* unmanagedMultislice;
		
	public:
		ManagedMultisliceSimulation();
		~ManagedMultisliceSimulation();
		void ImportAtoms(String^ filename);
		void UploadBinnedAtoms();
		void UploadBinnedAtomsTDS();
		void GetParameterisation();
		void ApplyMicroscopeParameters(float Voltage, float defocus, float Mod2f, float Arg2f, float spherical, float B, float D, float obj);
		void SetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float SizeX, float SizeY, float SizeZ, float blockoffsetx, float blockoffsety);
		void STEMSetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float SizeX, float SizeY, float SizeZ);
		void PreCalculateFrequencies();
		void InitialiseWavefunctions();
		void InitialiseWavefunctions(float dz);
		void STEMInitialiseWavefunctions(int posx, int posy);
		void MultisliceStepFD(int iteration);
		void MultisliceStepConventional(int iteration);

		float GetSizeX();
		float GetSizeY();
		float GetSizeZ();
		int GetSlices(bool conventional);
		float GetKmax();
		void GetExitWave();
		void GetSTEMExitWave();
		void AddTDSWaves();
		void AllocTDS();
		void FreeExitWave();
		void AllocHostImage();
		void SimulateImage(float doseperpix);
		float GetImageContrast(int t, int l, int b, int r);
		float GetEWValueAbs(int xpos, int ypos);
		float GetEWValueRe(int xpos, int ypos);
		float GetEWValueIm(int xpos, int ypos);
		float GetImValue(int xpos, int ypos);

		float GetImMin();
		float GetImMax();
		float GetEWMax();
		float GetEWMin();

		float GetTDSMin();
		float GetTDSMax();
		float GetTDSVal(int xpos, int ypos);
	};

	public ref class CUDAIterativeAdditionClass
	{

	private:
		unmanagedCUDAIterativeAdditionClass* unmanagedIterativeAddition;

	public:
		CUDAIterativeAdditionClass();
		~CUDAIterativeAdditionClass();

		
		int AddTwoNumbers(int numOne, int numTwo, int iteration, int maxIterations);

		void Alloc();
		void DeAlloc();

	};
}
	