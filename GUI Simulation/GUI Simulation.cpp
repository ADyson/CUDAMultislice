// This is the main DLL file.

#include "stdafx.h"
#include "GUI Simulation.h"



GUISimulation::CUDAIterativeAdditionClass::CUDAIterativeAdditionClass()
{
	unmanagedIterativeAddition = new unmanagedCUDAIterativeAdditionClass();
}

GUISimulation::CUDAIterativeAdditionClass::~CUDAIterativeAdditionClass()
{
	delete unmanagedIterativeAddition;
}

int GUISimulation::CUDAIterativeAdditionClass::AddTwoNumbers(int numOne, int numTwo, int iteration, int maxIterations)
{
	return(unmanagedIterativeAddition->AddTwoNumbers(numOne,numTwo,iteration,maxIterations));
}

void GUISimulation::CUDAIterativeAdditionClass::Alloc()
{
	unmanagedIterativeAddition->Alloc();
}

void GUISimulation::CUDAIterativeAdditionClass::DeAlloc()
{
	unmanagedIterativeAddition->DeAlloc();
}