#include "cuComplex.h"
#include <string>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#pragma unmanaged

struct Atom
{
	int atomicNumber;
	float x, y, z;
};

struct AtomParameterisation
{
	float a,b,c,d,e,f,g,h,i,j,k,l;
};

struct cudaMemories
	{
		int		*AtomZNum;
		float	*AtomXPos;
		float	*AtomYPos;
		float	*AtomZPos;

		int		*devAtomZNum;
		float	*devAtomXPos;
		float	*devAtomYPos;
		float	*devAtomZPos;
		float   *fparamsdev;

		int		*DBlockIds;
		int		*HBlockIds;
		int		*DZIds;
		int		*HZIds;
		int		*devBlockStartPositions;
	};



class unmanagedCUDAIterativeAdditionClass
{
	private:
		int* numOne;
		int* numTwo;
		int* result;

	public:
		unmanagedCUDAIterativeAdditionClass();
		~unmanagedCUDAIterativeAdditionClass();

		int AddTwoNumbers(int numOne, int numTwo, int iteration, int maxIterations);

		void Alloc();
		void DeAlloc();

};

class UnmanagedMultisliceSimulation 
{
	private:
		// Main arrays for CUDA Calculation
		cuComplex* PsiMinus;
		cuComplex* Psi;
		cuComplex* PsiPlus;
		float* xFrequencies;
		float* yFrequencies;
		cufftHandle plan;
		cufftHandle bigplan;
		cublasHandle_t cublashandle;
		cublasStatus_t cublasstatus;

		// Additional temp arrays used in calculation;
		cuComplex* devV;
		cuComplex* devGrad;
		cuComplex* devGrad2;
		cuComplex* devBandLimitStorage;

		// Arrays used for holding atoms;
		float* AtomXPos;
		float* AtomYPos;
		float* AtomZPos;
		int* AtomZNum;

		std::vector<Atom> AtomicStructure;
		cudaMemories atomMemories;

		// Variables for microscope parameters
		float df;
		float Mod2fold;
		float Arg2fold;
		float Mod3fold;
		float Arg3fold;
		float Cs;
		float beta;
		float delta;
		float kV;
		float objectiveAperture;

		// Variables for multislice calculation
		int resolutionX;
		int resolutionY;
		float PixelScale;
		float dzAtomSlice;
		float dzFiniteDifference;


		float blockxoffset;
		float blockyoffset;

		float ksizex;
		float ksizey;
		float ke2;
		float normalisingfactor;
		float wavel;
		float sigma2;

		//TODO: Put as private again and give access functions.
	public:
		UnmanagedMultisliceSimulation();
		~UnmanagedMultisliceSimulation();

		// Host Memory for Storing Atoms
		float MaximumX;
		float MaximumY;
		float MaximumZ;
		float MinimumX;
		float MinimumY;
		float MinimumZ;
		int numberOfSlices;
		float kmax;
		float conventionaldz;

		float maxposIm;
		float maxposEW;
		float minposEW;
		float minposIm;

		// Host Memory for ExitWave
		float2* EW;

		float* TDS;
		float* TDS2;

		// Host Memory for Simulated Image
		
		float2* SimulatedImage;
		
		

		void LoadAtomFile(std::string filepath);
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
		void AllocTDS();
		void MultisliceStep(int iteration);
		void MultisliceStepConv(int iteration);
		void MultisliceStepRS(int iteration);
		void AllocHostImage();
		void SimulateImage(float doseperpix);
		//void MonteCarloIteration();

		void GetExitWave();
		void GetSTEMExitWave();
		void FreeExitWave();

		void AddTDSWaves();

		//int GetEWMax();
		//int GetImMax();

		float GetEWValueAbs(int xpos, int ypos);
		float GetEWValueRe(int xpos, int ypos);
		float GetEWValueIm(int xpos, int ypos);
		float GetImValue(int xpos, int ypos);

		float GetTDSMin();
		float GetTDSMax();
		float GetTDSVal(int xpos, int ypos);

		float GetDQEImValue(int xpos, int ypos);
		//float GetImPhaseValue(int xpos, int ypos);
		//float GetDiffValue(int xpos, int ypos);

		float GetImageContrast(int t, int b, int l, int r);

		float imagecontrast;

	
};