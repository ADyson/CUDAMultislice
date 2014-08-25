#include "Stdafx.h"
#include "unmanagedGUISimulation.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include "math.h"
#include "cuda.h"
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>



#include "MTF.h"
using namespace std;

// CUDA Kernels

__device__ __constant__ int consts[3]; 
__device__ __constant__ float maxs[6];
__device__ __constant__ int res[2];
__device__ __constant__ int blocks[2];
__device__ __constant__ int grids[2];
__device__ __constant__ int bins[2];
__device__ __constant__ float temParams[10];

// Generate randomised atom positions for TDS
float TDSRand()
{
	double random = ((double) rand() / (RAND_MAX+1));
	double random2 = ((double) rand() / (RAND_MAX+1));
	double rstdnormal = sqrt(-2.0f * +log(FLT_MIN+random))*(sin(2.0f * 3.1415926f * random2));
	float randNormal = 0.075f * rstdnormal; //random normal(mean,stdDev^2)

	return randNormal;
}

// Device function to calculation Bessel0

__device__ float bessi0( float x )
 {

	int i;
	float ax, sum, t;

	float i0a[] = { 1.0, 3.5156229, 3.0899424, 1.2067492,
		0.2659732, 0.0360768, 0.0045813 };

	float i0b[] = { 0.39894228, 0.01328592, 0.00225319,
		-0.00157565, 0.00916281, -0.02057706, 0.02635537,
		-0.01647633, 0.00392377};

	ax = fabs( x );

	if( ax <= 3.75 ) 
	{
		t = x / 3.75;
		t = t * t;
		sum = i0a[6];

		for( i=5; i>=0; i--) sum = sum*t + i0a[i]; 

	} else
	{
		t = 3.75 / ax;
		sum = i0b[8];
		for( i=7; i>=0; i--) sum = sum*t + i0b[i];
		sum = exp( ax ) * sum / sqrt( ax );
	}

	return( sum );
}

// Device function to calculate BesselK

__device__ float bessk0(float x) 
{
	
	float bessi0(float);

	int i;
	float ax, x2, sum;
	float k0a[] = { -0.57721566, 0.42278420, 0.23069756, 0.03488590, \
		0.00262698, 0.00010750, 0.00000740};
	float k0b[] = { 1.25331414, -0.07832358, 0.02189568, -0.01062446, \
		0.00587872, -0.00251540, 0.00053208};
		
	ax = fabs( x );

	if( (ax > 0.0)  && ( ax <=  2.0 ) )
	{
		x2 = ax/2.0;
		x2 = x2 * x2;
		sum = k0a[6];
		for( i=5; i>=0; i--) sum = sum*x2 + k0a[i];
		sum = -log(ax/2.0) * bessi0(x) + sum;

	} else if( ax > 2.0 ) 
	{
		x2 = 2.0/ax;
		sum = k0b[6];

		for( i=5; i>=0; i--) sum = sum*x2 + k0b[i];

		sum = exp( -ax ) * sum / sqrt( ax );

	} else sum = 1.0e20;

	return ( sum );
}

///////////////////////////////////////////////////////////////////////////
// VzAtom2 - Device function to calculate projected potential at r from  //
// an atom with atomic number Z, using input parameters devfparams.      //
// NB. input should be the radius^2 , it is sqrt in this function.       //
///////////////////////////////////////////////////////////////////////////

__device__ float vzatom2( int Z, float radius,const float * devfparams)
{

	int i;
	float suml, sumg, x, r;

	/* Lorenzian, Gaussian consts */

	r = fabs( radius);
	r= sqrt(r);

	if( r < 0.25 ) r = 0.25f; // was 0.15

	/* avoid singularity at r=0 */

   suml = sumg = 0.0;

   /* Lorenztians */

   x = 2.0f*3.141592654f*r;

   for( i=0; i<2*3; i+=2 )
		suml += devfparams[(Z-1)*12+i]* bessk0( x*sqrt(devfparams[(Z-1)*12+i+1]) );
   
   /* Gaussians */

   x = 3.141592654f*r;
   x = x*x;

   for( i=2*3; i<2*(3+3); i+=2 )
		sumg += devfparams[(Z-1)*12+i] * exp (-x/devfparams[(Z-1)*12+i+1]) / devfparams[(Z-1)*12+i+1];

   return( 300.8242834f*suml + 150.4121417f*sumg );
} 

///////////////////////////////////////////////////////////////////////////
// vatom - Device function to calculate potential at a distance r from //
// an atom with atomic number Z, using input parameters devfparams.      //
// NB. input should be the radius^2 , it is sqrt in this function.       //
///////////////////////////////////////////////////////////////////////////

__device__ float vatom( int Z, float radius,const float * devfparams)
{

	int i;
	float suml, sumg, x, r;

	/* Lorenzian, Gaussian consts */

	r = fabs( radius);
	r= sqrt(r);

	if( r < 0.25 ) r = 0.25f;

	/* avoid singularity at r=0 */

   suml = sumg = 0.0;

   /* Lorenztians */

   x = 2.0*3.141592654f*r;

   for( i=0; i<2*3; i+=2 )
		suml += (1/r) * devfparams[(Z-1)*12+i]* exp( -x*sqrt(devfparams[(Z-1)*12+i+1]) );
   
   /* Gaussians */

   x = 3.141592654f*r;
   x = x*x;

   for( i=2*3; i<2*(3+3); i+=2 )
		sumg += devfparams[(Z-1)*12+i] * exp (-x/devfparams[(Z-1)*12+i+1]) * powf(devfparams[(Z-1)*12+i+1],-1.5);

   return( 150.4121417f*suml + 266.5157269f*sumg );
} 



__global__ void conjugationKernel( cuComplex * PsiIn, float * PsiOut, int mode, int diffpattern)
{
	
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int xIndex2 = xIndex-res[0]/2;
	int yIndex2 = yIndex-res[1]/2;


	if(xIndex<res[0]/2)
	{
		xIndex2=(xIndex+res[0]/2);
	}

	if(yIndex<res[1]/2)
	{
		yIndex2=(yIndex+res[1]/2);
	}

	
	int Index = (yIndex * res[0] + xIndex);
	int Index2 = ((res[1]-yIndex)*res[0] + (res[0]-xIndex));


	int Index3 = (yIndex2 * res[0] + xIndex2);
	int Index4 = ((res[1]-1-yIndex2)*res[0] + (res[0]-1-xIndex2));


	if (diffpattern==1){
		PsiOut[Index] = log(1+3000*((PsiIn[Index3].x + PsiIn[Index4].x)*(PsiIn[Index3].x + PsiIn[Index4].x) + (PsiIn[Index3].y - PsiIn[Index4].y)*(PsiIn[Index3].y - PsiIn[Index4].y)));
		}
	else if(mode==1)
		PsiOut[Index] = sqrtf((PsiIn[Index].x * PsiIn[Index].x) + (PsiIn[Index].y * PsiIn[Index].y));
	else if(mode==2)
		PsiOut[Index] = PsiIn[Index].x;
	else if(mode==3)
		PsiOut[Index] = PsiIn[Index].y;
	
}

__global__ void add_kernel(int* numOne, int* numTwo, int* result) {

	int tidx = threadIdx.x + blockIdx.x * blockDim.x;

	if(tidx>=1)
		exit;

	result[tidx] = numOne[tidx] + numTwo[tidx];
}

__global__ void atombin(float * xpos, float * ypos, int length, float maxx, float minx, float maxy, 
	float miny, int xBlocks , int yBlocks, int * bids, int * zids, float * zpos, float maxz, float dz,int nSlices) 
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (xIndex >= length)
		return ;

	int bidx = floor((xpos[xIndex] - minx)/(maxx-minx)*xBlocks);
	int bidy = floor((ypos[xIndex] - miny)/(maxy-miny)*yBlocks);

	int zid  = floor((maxz-zpos[xIndex])/dz);
	zid-=(zid==nSlices);

	bidx-=(bidx==xBlocks);
	bidy-=(bidy==yBlocks);

	int bid = bidx + xBlocks*bidy;

	bids[xIndex] = bid;
	zids[xIndex] = zid;
	
}

__global__ void atombin2(float * xpos, float * ypos, int start, int length, float maxx, float minx, float maxy, 
	float miny, int xBlocks , int yBlocks, int * bids, int * zids, float * zpos, float maxz, float dz,int nSlices) 
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (xIndex >= length)
		return ;

	int Index = start + xIndex;

	int bidx = floor((xpos[Index] - minx)/(maxx-minx)*xBlocks);
	int bidy = floor((ypos[Index] - miny)/(maxy-miny)*yBlocks);

	int zid  = floor((maxz-zpos[Index])/dz);
	
	// This shouldn't ever happen because of the extra space around the edge of model.
	//zid-=(zid==nSlices);
	//bidx-=(bidx==xBlocks);
	//bidy-=(bidy==yBlocks);

	int bid = bidx + xBlocks*bidy;

	bids[Index] = bid;
	zids[Index] = zid;
	
}

__global__ void initializingKernel(cuComplex * in1, float value)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);
	
	in1[Index].x 	= value;
	in1[Index].y 	= 0.0f;
	
  }


__global__ void makeComplex(float* in, cuComplex * in1)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);
	
	in1[Index].x 	= in[Index];
	in1[Index].y 	= 0.0f;
	
  }

__global__ void makeComplexSq(float* in, cuComplex * in1)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);
	
	in1[Index].x 	= in[Index]*in[Index];
	in1[Index].y 	= 0.0f;
	
  }

__global__ void STEMinitializingKernel(cuComplex * in1, float* xFrequencies, float* yFrequencies, int posx, int posy, float apert, float pixelscale, float df, float Cs, float wavel)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);

	float k0x = xFrequencies[xIndex];
	float k0y = yFrequencies[yIndex];
	float k = sqrtf(k0x*k0x + k0y*k0y) ;
	float Pi = 3.14159265f;
	
	if( k < apert)
	{
		//in1[Index].x 	= cosf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df))*cosf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale));
		//in1[Index].y 	= -sinf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df))*sinf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale));
		in1[Index].x 	= cosf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df))*cosf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale))  + sinf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df))*sinf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale)) ;
		in1[Index].y 	= -cosf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale))*sinf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df)) + cosf(Pi*wavel*k*k*(Cs*wavel*wavel*k*k*0.5f + df))*sinf(2*Pi*(k0x*posx*pixelscale + k0y*posy*pixelscale));
	}
	else
	{
		in1[Index].x 	= 0.0f;
		in1[Index].y 	= 0.0f;
	}
	
  }

__global__ void multiplicationKernel(cuComplex * in1, cuComplex * in2, cuComplex * out,float normalise)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);
	
	out[Index].x 	= (in1[Index].x * in2 [Index].x - in1 [Index].y * in2 [Index].y)*normalise;
	out[Index].y 	= (in1[Index].x * in2 [Index].y + in1 [Index].y * in2 [Index].x)*normalise;	
  }

__global__ void CreatePropsKernel( float * kxIn, float * kyIn, float dz, float wavel, 
	cuComplex * Props, float kmax)
{
	
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = yIndex * res[0] + xIndex;

	float k0x = kxIn[xIndex];
	float k0y = kyIn[yIndex];
	float Pi = 3.14159265f;

	k0x*=k0x;
	k0y*=k0y;

	if (sqrtf(k0x+k0y) < kmax)
	{
		Props[Index].x = cosf(Pi*dz*wavel*(k0x+k0y));
		Props[Index].y = -1*sinf(Pi*dz*wavel*(k0x+k0y));
	} else 
	{
		Props[Index].x = 0.0f;
		Props[Index].y = 0.0f;
	}
}

__global__ void normalisingKernel(cuComplex * in, float normalisingFactor)
{
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	in[Index].x *= normalisingFactor;
	in[Index].y *= normalisingFactor;
}

__global__ void binnedAtomPKernel(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, int dz,
	float z, int nSlices,float sigma)
{

	float vzatom2(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz);
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz);
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float((blockIdx.y * blocks[1] * bins[1] * pixelscale/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockIdx.y+1) * blocks[1] * bins[1] * pixelscale/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float((blockIdx.x * bins[0] * blocks[0] * pixelscale / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockIdx.x+1) * bins[0] * blocks[0] * pixelscale/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
									rad2 = (xIndex*pixelscale-devAtomXPos[l])*(xIndex*pixelscale-devAtomXPos[l]) + (yIndex*pixelscale-devAtomYPos[l])*(yIndex*pixelscale-devAtomYPos[l]);
																
									if( rad2 < 5.0f) // Check atom is within specified range.
										sumz += vzatom2(devAtomZNum[l] , rad2 , devfparams);
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = cosf(sigma*sumz);
	V[Index].y = sinf(sigma*sumz);
}



// Attempted to add an offset so that the simulation will always be central when I change mag..
__global__ void binnedAtomPKernel2(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, int dz,
	float z, int nSlices,float sigma, float blockoffsetx, float blockoffsety)
{

	float vzatom2(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz);
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz);
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	// Could do offset in terms of number of blocks?

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float(((blockoffsety+blockIdx.y * blocks[1] * pixelscale)* bins[1]/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockoffsety+(blockIdx.y+1) * blocks[1]  * pixelscale)* bins[1]/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float(((blockoffsetx+blockIdx.x  * blocks[0] * pixelscale)* bins[0] / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockoffsetx+(blockIdx.x+1)  * blocks[0] * pixelscale)* bins[0]/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
									rad2 = (blockoffsetx+xIndex*pixelscale-devAtomXPos[l])*(blockoffsetx+xIndex*pixelscale-devAtomXPos[l]) + (blockoffsety+yIndex*pixelscale-devAtomYPos[l])*(blockoffsety+yIndex*pixelscale-devAtomYPos[l]);
																
									if( rad2 < 5.0f) // Check atom is within specified range.
										sumz += vzatom2(devAtomZNum[l] , rad2 , devfparams);
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = cosf(sigma*sumz);
	V[Index].y = sinf(sigma*sumz);
}


// Attempted to add an offset so that the simulation will always be central when I change mag..
__global__ void binnedAtomPKernel3(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, int dz, float convdz,
	float z, int nSlices,float sigma, float blockoffsetx, float blockoffsety)
{

	float vatom(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz)-consts[2];
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz)+consts[2];
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	// Could do offset in terms of number of blocks?

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float(((blockoffsety+blockIdx.y * blocks[1] * pixelscale)* bins[1]/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockoffsety+(blockIdx.y+1) * blocks[1]  * pixelscale)* bins[1]/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float(((blockoffsetx+blockIdx.x  * blocks[0] * pixelscale)* bins[0] / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockoffsetx+(blockIdx.x+1)  * blocks[0] * pixelscale)* bins[0]/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
									
									for(int m = 1; m <= 20; m++)
									{
										float z2 = z -m*convdz/20.0f;
										rad2 = (blockoffsetx+xIndex*pixelscale-devAtomXPos[l])*(blockoffsetx+xIndex*pixelscale-devAtomXPos[l]) + (blockoffsety+yIndex*pixelscale-devAtomYPos[l])*(blockoffsety+yIndex*pixelscale-   
												devAtomYPos[l]) + (z2-devAtomZPos[l])*(z2-devAtomZPos[l]);
																
									if( rad2 < 5.0f) // Check atom is within specified range.
										sumz += convdz/20.0f * vatom(devAtomZNum[l] , rad2 , devfparams);
									}
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = cosf(sigma*sumz);
	V[Index].y = sinf(sigma*sumz);
}

// Attempted to add an offset so that the simulation will always be central when I change mag..
__global__ void binnedAtomPKernelRS(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, int dz,
	float z, int nSlices,float sigma, float blockoffsetx, float blockoffsety)
{

	float vatom(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz)-consts[2];
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz)+consts[2];
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	// Could do offset in terms of number of blocks?

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float(((blockoffsety+blockIdx.y * blocks[1] * pixelscale)* bins[1]/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockoffsety+(blockIdx.y+1) * blocks[1]  * pixelscale)* bins[1]/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float(((blockoffsetx+blockIdx.x  * blocks[0] * pixelscale)* bins[0] / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockoffsetx+(blockIdx.x+1)  * blocks[0] * pixelscale)* bins[0]/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
									//Number of steps in integration....
									for(int m = 1; m <= 10; m++)
									{
										float z2 = z -m*dz/10.0f;
										rad2 = (blockoffsetx+xIndex*pixelscale-devAtomXPos[l])*(blockoffsetx+xIndex*pixelscale-devAtomXPos[l]) + (blockoffsety+yIndex*pixelscale-devAtomYPos[l])*(blockoffsety
											+yIndex*pixelscale-	devAtomYPos[l]) + (z2-devAtomZPos[l])*(z2-devAtomZPos[l]);
																
										if( rad2 < 5.0f) // Check atom is within specified range.
											sumz += dz/10.0f * vatom(devAtomZNum[l] , rad2 , devfparams);
									}
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = sumz;
	V[Index].y = 0;
}

// Attempted to add an offset so that the simulation will always be central when I change mag..
__global__ void binnedAtomPKernelConventional3(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, float dz,
	float z, int nSlices,float sigma, float blockoffsetx, float blockoffsety)
{

	float vzatom2(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz); // Only atoms in slice...
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz);
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	// Could do offset in terms of number of blocks?

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float(((blockoffsety+blockIdx.y * blocks[1] * pixelscale)* bins[1]/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockoffsety+(blockIdx.y+1) * blocks[1]  * pixelscale)* bins[1]/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float(((blockoffsetx+blockIdx.x  * blocks[0] * pixelscale)* bins[0] / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockoffsetx+(blockIdx.x+1)  * blocks[0] * pixelscale)* bins[0]/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
								rad2 = (blockoffsetx+xIndex*pixelscale-devAtomXPos[l])*(blockoffsetx+xIndex*pixelscale-devAtomXPos[l]) + (blockoffsety+yIndex*pixelscale-devAtomYPos[l])*(blockoffsety+yIndex*pixelscale-devAtomYPos[l]);
																
									if( rad2 < 5.0f) // Check atom is within specified range.
										sumz += vzatom2(devAtomZNum[l] , rad2 , devfparams);
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = cosf(sigma*sumz);
	V[Index].y = sinf(sigma*sumz);
}

// 5 point double differential
__global__ void DifferentialXKernel(cuComplex * in, cuComplex * out, float pixelscale)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);

	int p1 = (xIndex>0)*(xIndex-1);
	int p2 = (xIndex>1)*(xIndex-2);
	int n1 = (xIndex<res[0]-1)*(xIndex+1);
	int n2 = (xIndex<res[0]-2)*(xIndex+2);

	float d1x = in[n1+res[0]*yIndex].x + in[p1+res[0]*yIndex].x - in[Index].x;
	float d2x = in[n2+res[0]*yIndex].x + in[p2+res[0]*yIndex].x - in[Index].x;
	float d1y = in[n1+res[0]*yIndex].y + in[p1+res[0]*yIndex].y - in[Index].y;
	float d2y = in[n2+res[0]*yIndex].y + in[p2+res[0]*yIndex].y - in[Index].y;
	
	out[Index].x 	= (1/(pixelscale*pixelscale))*(4.0f* d1x/3.0f -d2x/12.0f);
	out[Index].y 	= (1/(pixelscale*pixelscale))*(4.0f* d1y/3.0f -d2y/12.0f);
  }

__global__ void DifferentialYKernel(cuComplex * in, cuComplex * out, float pixelscale)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);

	int p1 = (yIndex>0)*(yIndex-1);
	int p2 = (yIndex>1)*(yIndex-2);
	int n1 = (yIndex<res[1]-1)*(yIndex+1);
	int n2 = (yIndex<res[1]-2)*(yIndex+2);

	float d1x = in[xIndex+res[0]*n1].x + in[xIndex+res[0]*p1].x - in[Index].x;
	float d2x = in[xIndex+res[0]*n2].x + in[xIndex+res[0]*p2].x - in[Index].x;
	float d1y = in[xIndex+res[0]*n1].y + in[xIndex+res[0]*p1].y - in[Index].y;
	float d2y = in[xIndex+res[0]*n2].y + in[xIndex+res[0]*p2].y - in[Index].y;
	
	out[Index].x 	= (1/(pixelscale*pixelscale))*(4.0f* d1x/3.0f -d2x/12.0f);
	out[Index].y 	= (1/(pixelscale*pixelscale))*(4.0f* d1y/3.0f -d2y/12.0f);
  }

__global__ void RSMS(cuComplex * dxin, cuComplex * dyin, cuComplex * wavein, cuComplex* potentialin, cuComplex * out, float wavel, float dz, int m)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0] + xIndex);
	float pre = (1.0f/m)*(wavel*dz/(4.0f*3.1415927f));

	float r1x;
	float r1y;

	if(m==1)
	{
		wavein[Index].x = 1.0f;
		wavein[Index].y=0.0f;

		r1x = 0;
		r1y = pre * potentialin[Index].x;
	}
	else
	{
		r1x = -1.0f * pre * (dxin[Index].y + dyin[Index].y + potentialin[Index].x * wavein[Index].y );
		r1y = pre * (dxin[Index].x + dyin[Index].x + potentialin[Index].x * wavein[Index].x );
	}
	wavein[Index].x 	= wavein[Index].x + r1x;
	wavein[Index].y 	= wavein[Index].y + r1y;
	//out[Index].x 	= potentialin[Index].x;
	//out[Index].y 	= potentialin[Index].x;
  }





__global__ void binnedAtomPKernelFD(cuComplex * V, float pixelscale, const float * devfparams,
	float * devAtomZPos, float * devAtomXPos, float * devAtomYPos, int * devAtomZNum, int * devBlockStartPositions, int dz,
	float z, int nSlices)
{

	float vatom(int Z, float radius, const float * devfparams);

	float sumz = 0; // To store value of potential, initialized to zero for call of each slice.
	float rad2 = 0;
		
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;
	
	// Not entirely certain about this part.
	int topzid  = floor((maxs[4]-maxs[5]-z)/dz) - consts[2];
	int bottomzid = floor((maxs[4]-maxs[5]-z)/dz) + consts[2];
	if (topzid < 0) topzid = 0;
	if (bottomzid >= nSlices) bottomzid = nSlices-1;


	// NB - consts[] contains number of blocks and slices to loads in constant memory, 0=x,1=y,2=z.


	// The next two lines calculate which atom blockID's to load based on this threads blockID, complicated conversion is required because
	// the number of blocks in each direction (atoms) does not correlate with the number of blocks in each direction (threads). Also the
	// area covered by each block takes into account the the blocks include extra threads which overhang the end of the sample if it does
	// not perfectly divide by the block size. This should scale correctly if the number of blocks is ever changed :)

	for(int k = topzid; k <= bottomzid; k++)
	{
		for (int j = floor(float((blockIdx.y * blocks[1] * bins[1] * pixelscale/ ( maxs[2]-maxs[3] ))) - consts[1] ); j <= ceil(float(((blockIdx.y+1) * blocks[1] * bins[1] * pixelscale/ ( maxs[2] - maxs[3] ))) + consts[1]); j++)
		{
			for (int i = floor(float((blockIdx.x * bins[0] * blocks[0] * pixelscale / (maxs[0]-maxs[1] ))) - consts[0] ); i <= ceil(float(((blockIdx.x+1) * bins[0] * blocks[0] * pixelscale/ ( maxs[0] - maxs[1] ))) + consts[0]); i++)
			{
				// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
				if(0 <= j && j < bins[1]) 
				{
					// Check bounds to avoid unneccessarily loading blocks when i am at edge of sample.
					if (0 <= i && i < bins[0] ) 
						{	
							// Check if there is an atom in bin, arrays are not overwritten when there are no extra atoms so if you don't check could add contribution more than once.
							for (int l = devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i]; l<devBlockStartPositions[k*bins[0]*bins[1] + bins[0]*j + i+1]; l++) 
							{
									rad2 = (xIndex*pixelscale-devAtomXPos[l])*(xIndex*pixelscale-devAtomXPos[l]) + (yIndex*pixelscale-devAtomYPos[l])*(yIndex*pixelscale-devAtomYPos[l]) + (z-devAtomZPos[l])*(z-devAtomZPos[l]);
																
									if( rad2 < 5.0f) // Check atom is within specified range.
										sumz += vatom(devAtomZNum[l] , rad2 , devfparams);
										
							}
						}
				}
			}
		}
	}
	
	V[Index].x = sumz;
	V[Index].y = 0;

}

__global__ void gradKernel(cuComplex * in,float * k0xdev, float * k0ydev, float normalisingFactor)
{
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	in[Index].x = in[Index].x * normalisingFactor * -4 * 3.14159f * 3.14159f * ((k0xdev[xIndex] * k0xdev[xIndex]) + (k0ydev[yIndex] * k0ydev[yIndex])) ;
	in[Index].y = in[Index].y * normalisingFactor * -4 * 3.14159f * 3.14159f * ((k0xdev[xIndex] * k0xdev[xIndex]) + (k0ydev[yIndex] * k0ydev[yIndex])) ;

}

__global__ void CentreKernel(cuComplex * in,cuComplex * out)
{
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;

	int xIndex2 = xIndex + res[0]/2;
	int yIndex2 = yIndex + res[1]/2;
		
	int Index = (yIndex * res[0]) + xIndex;
	
	if(xIndex >= (res[0]/2))
	{
		xIndex2 = xIndex - res[0]/2;
	}
	
	if(yIndex >= (res[1]/2))
	{
		yIndex2 = yIndex - res[1]/2;
	}

	int Index2 = xIndex2 + res[0]*yIndex2;
	
	out[Index].x = in[Index2].x;
	out[Index].y = in[Index2].y;

}

__global__ void finiteDifferenceKernel(cuComplex * devGrad, cuComplex * devPsiMinusDz, cuComplex * devPsiPlusDz, cuComplex * devPsiZ, cuComplex * devV, float sigma, float wavel,float FDdz)
{
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = (yIndex * res[0]) + xIndex;

	cuComplex cMinus = {1 , -2*3.14159f*FDdz/wavel}; 
	cuComplex cPlus = {1 , 2*3.14159f*FDdz/wavel}; 

	cuComplex reciprocalCPlus = {cMinus.x / (cMinus.x*cMinus.x + cMinus.y*cMinus.y),cMinus.y / (cMinus.x*cMinus.x + cMinus.y*cMinus.y)};
	cuComplex cMinusOvercPlus = {(cPlus.x*cPlus.x - cPlus.y*cPlus.y) / (cMinus.x*cMinus.x + cMinus.y*cMinus.y),-2*(cPlus.x*cPlus.y) / (cMinus.x*cMinus.x + cMinus.y*cMinus.y)};

	float real = reciprocalCPlus.x*(2*devPsiZ[Index].x-FDdz*FDdz*devGrad[Index].x - FDdz*FDdz*4*3.14159f*sigma*devV[Index].x*devPsiZ[Index].x/wavel)
				-reciprocalCPlus.y*(2*devPsiZ[Index].y-FDdz*FDdz*devGrad[Index].y -  FDdz*FDdz*4*3.14159f*sigma*devV[Index].x*devPsiZ[Index].y/wavel)
				-cMinusOvercPlus.x*(devPsiMinusDz[Index].x) + cMinusOvercPlus.y*(devPsiMinusDz[Index].y);

	float imag = reciprocalCPlus.y*(2*devPsiZ[Index].x-FDdz*FDdz*devGrad[Index].x - FDdz*FDdz*4*3.14159f*sigma*devV[Index].x*devPsiZ[Index].x/wavel)
				+reciprocalCPlus.x*(2*devPsiZ[Index].y-FDdz*FDdz*devGrad[Index].y -  FDdz*FDdz*4*3.14159f*sigma*devV[Index].x*devPsiZ[Index].y/wavel)
				-cMinusOvercPlus.y*(devPsiMinusDz[Index].x) - cMinusOvercPlus.x*(devPsiMinusDz[Index].y);

	devPsiPlusDz[Index].x = real;
	devPsiPlusDz[Index].y = imag;

}

__global__ void BandLimitKernel( cuComplex * FT, float kmax, float * k0xdev, float * k0ydev)
{
	
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= res[0] || yIndex >= res[1])
		return ;
		
	int Index = yIndex * res[0] + xIndex;

	float k = sqrt(k0xdev[xIndex]*k0xdev[xIndex] + k0ydev[yIndex]*k0ydev[yIndex]);

	FT[Index].x = (k <= kmax)*FT[Index].x;
	FT[Index].y = (k <= kmax)*FT[Index].y;

}

__global__ void imagingKernel2(cuComplex * PsiIn, int samplex, int sampley, float Cs, float df, float dfa2, 
	float dfa2phi, float dfa3,float dfa3phi,float objap,float wavel, cuComplex * PsiOut, float * k0x, float * k0y,
	float beta, float delta)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= samplex || yIndex >= sampley)
		return ;

	int Index = (yIndex * samplex + xIndex);
	
	float objap2 = (((objap * 0.001f) / wavel ) * (( objap * 0.001f ) / wavel ));
	float Chi1 = 3.1415926f * wavel;
	float Chi2 = 0.5f * Cs * wavel * wavel;
	float k2 = (k0x[xIndex]*k0x[xIndex]) + (k0y[yIndex]*k0y[yIndex]);
	float k =sqrtf(k2);
	float factor = 1.0f*beta*beta/(4*wavel*wavel);

	float ecohs = expf(-factor*pow(3.141593f*k*wavel*2*df + 2*3.141593f*wavel*wavel*wavel*Cs*k2*k,2));
	float ecohd = expf(-0.25f*delta*delta*3.141593f*3.141593f*k2*k2*wavel*wavel);
	
	if ( k2 < objap2){
	
		float Phi = atan2(k0y[yIndex],k0x[xIndex]);
		float Chi = Chi1 * k2 * ( Chi2 * k2 + df + dfa2 * sin ( ( 2.0f * ( Phi - dfa2phi ) ) ) + 2.0f * dfa3 * wavel * sqrtf ( k2 ) * sinf ( ( 3.0f * ( Phi - dfa3phi ) ) ) / 3.0f );
		PsiOut[Index].x = ecohs*ecohd*(PsiIn[Index].x *  cosf ( Chi )  + PsiIn[Index].y * sinf ( Chi )) ;
		PsiOut[Index].y	= ecohs*ecohd*(PsiIn[Index].x * -1 *  sinf ( Chi ) + PsiIn[Index].y * cosf ( Chi ));
	}
	else {
	
		PsiOut[Index].x = 0.0f;
		PsiOut[Index].y = 0.0f;
	}

}

__global__ void SubImageKernel( float * imagein, float * subimageout, int t, int l , int b , int r, int bigwidth, int bigheight)
{
	
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= (r - l) || yIndex >= (b - t))
		return ;
		
	int Index = yIndex * (r - l) + xIndex;

	int xloc = l + xIndex;
	int yloc = t + yIndex;

	int loc = yloc * bigwidth + xloc;

	subimageout[Index] = imagein[loc];

}

__global__ void StdDevKernel( float * imagein, int width, int height, float average)
{
	
	
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (xIndex >= width || yIndex >= height)
		return ;
		
	int Index = yIndex * width + xIndex;


	imagein[Index] = pow((imagein[Index]-average),2);

}

__device__ float DQEFunc( int xdis, int ydis, int bin, float* dqe )
{
	float rad = sqrtf(xdis*xdis + ydis*ydis)/bin;
	int rad2 = floor(rad); // change to interpolated.
	if(rad2 <= 724)
	{
		return dqe[rad2];
	}
	else
	{
		return 0.0001f;
	}
}

__global__ void DQEKernel(cuComplex * Image, float* DQE, int samplex, int sampley, cuComplex * PsiOut, int binning)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= samplex || yIndex >= sampley)
		return ;

	int Index = (yIndex * samplex + xIndex);
	int midx;
	int midy;
	if(xIndex < samplex/2)
		midx=0;
	else
		midx=samplex;
	if(yIndex < sampley/2)
		midy=0;
	else
		midy=sampley;

	float DQEVal = DQEFunc(xIndex-midx,yIndex-midy,binning,DQE);

	PsiOut[Index].x = Image[Index].x * sqrt(DQEVal);
	PsiOut[Index].y = Image[Index].y * sqrt(DQEVal);
}

__global__ void NTFKernel(cuComplex * Image, float* DQE, int samplex, int sampley, cuComplex * PsiOut, int binning)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= samplex || yIndex >= sampley)
		return ;

	int Index = (yIndex * samplex + xIndex);
	int midx;
	int midy;
	if(xIndex < samplex/2)
		midx=0;
	else
		midx=samplex;
	if(yIndex < sampley/2)
		midy=0;
	else
		midy=sampley;

	float DQEVal = DQEFunc(xIndex-midx,yIndex-midy,binning,DQE);

	PsiOut[Index].x = Image[Index].x * DQEVal;
	PsiOut[Index].y = Image[Index].y * DQEVal;
}

int GetZNum(std::string atomSymbol)
{
	if (atomSymbol == "H")
		return 1;
	else if (atomSymbol == "He")
		return 2;
	else if (atomSymbol == "Li")
		return 3;
	else if (atomSymbol == "Be")
		return 4;
	else if (atomSymbol == "B")
		return 5;
	else if (atomSymbol == "C")
		return 6;
	else if (atomSymbol == "N")
		return 7;
	else if (atomSymbol == "O")
		return 8;
	else if (atomSymbol == "F")
		return 9;
	else if (atomSymbol == "Na")
		return 11;
	else if (atomSymbol == "Mg")
		return 12;
	else if (atomSymbol == "Si")
		return 14;
	else if (atomSymbol == "P")
		return 15;
	else if (atomSymbol == "S")
		return 16;
	else if (atomSymbol == "Cl")
		return 17;
	else if (atomSymbol == "Ca")
		return 20;
	else if (atomSymbol == "Cr")
		return 24;
	else if (atomSymbol == "Br")
		return 35;
	else if (atomSymbol == "Fe")
		return 26;
	else if (atomSymbol == "Sr")
		return 38;
	else if (atomSymbol == "Ru")
		return 44;
	else if (atomSymbol == "La")
		return 57;
	else if (atomSymbol == "Sm")
		return 62;
	else if (atomSymbol == "Ta")
		return 73;
	else if (atomSymbol == "W")
		return 74;

	else return 1;

}



unmanagedCUDAIterativeAdditionClass::unmanagedCUDAIterativeAdditionClass()
{
}

unmanagedCUDAIterativeAdditionClass::~unmanagedCUDAIterativeAdditionClass()
{
}

int unmanagedCUDAIterativeAdditionClass::AddTwoNumbers(int numOneHost, int numTwoHost, int iteration, int maxIterations)
{
	
	cudaMemcpy(numOne,&numOneHost,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(numTwo,&numTwoHost,sizeof(int),cudaMemcpyHostToDevice);

	add_kernel<<<1,1>>>(numOne,numTwo,result);

	int* returnval;

	cudaMallocHost(&returnval,sizeof(int));
	cudaMemcpy(returnval,result,sizeof(int),cudaMemcpyDeviceToHost);
	
	int returnResult = returnval[0];
	cudaFreeHost(returnval);
	
	return  returnResult;
}

void unmanagedCUDAIterativeAdditionClass::Alloc()
{
	cudaMalloc(&numOne,sizeof(int));
	cudaMalloc(&numTwo,sizeof(int));
	cudaMalloc(&result,sizeof(int));
}

void unmanagedCUDAIterativeAdditionClass::DeAlloc()
{
	cudaFree(numOne);
	cudaFree(numTwo);
	cudaFree(result);
}






void UnmanagedMultisliceSimulation::LoadAtomFile(std::string filepath)
{
	std::ifstream inputFile(filepath,std::ifstream::in);
	//inputFile.open(filename,ios::in);

	Atom linebuffer;

	if (!inputFile) 
	{
		// TODO: Not sure what do to in error situation :(
	}


	int numAtoms;
	std::string commentline;

	inputFile >> numAtoms;
	getline(inputFile,commentline);

	for(int i=1; i<= numAtoms; i++)
	{
		std::string atomSymbol;

		inputFile >> atomSymbol >> linebuffer.x >> linebuffer.y >> linebuffer.z;

		linebuffer.atomicNumber = GetZNum(atomSymbol);

		AtomicStructure.push_back (linebuffer);
	}

	inputFile.close();

	// Find Structure Range Also
	int maxX(0);
	int minX(0);
	int maxY(0);
	int minY(0);
	int maxZ(0);
	int minZ(0);

	for(int i = 1; i < AtomicStructure.size(); i++)
	{
	
		if (AtomicStructure[i].x > AtomicStructure[maxX].x)
			maxX=i;
		if (AtomicStructure[i].y > AtomicStructure[maxY].y)
			maxY=i;
		if (AtomicStructure[i].z > AtomicStructure[maxZ].z)
			maxZ=i;
		if (AtomicStructure[i].x < AtomicStructure[minX].x)
			minX=i;
		if (AtomicStructure[i].y < AtomicStructure[minY].y)
			minY=i;
		if (AtomicStructure[i].z < AtomicStructure[minZ].z)
			minZ=i;
	};

	MaximumX = AtomicStructure[maxX].x+2;
	MinimumX = AtomicStructure[minX].x-2;
	MaximumY = AtomicStructure[maxY].y+2;
	MinimumY = AtomicStructure[minY].y-2;
	MaximumZ = AtomicStructure[maxZ].z+2;
	MinimumZ = AtomicStructure[minZ].z-2;	
}

void UnmanagedMultisliceSimulation::UploadBinnedAtoms()
{
	// Currently has problems importing some very large structures but not sure which part is causing the problem without proper access to debugging.
	atomMemories.AtomZNum = new int[AtomicStructure.size()];
	atomMemories.AtomXPos = new float[AtomicStructure.size()];
	atomMemories.AtomYPos = new float[AtomicStructure.size()];
	atomMemories.AtomZPos = new float[AtomicStructure.size()];

	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		*(atomMemories.AtomZNum + i) = AtomicStructure[i].atomicNumber;
		*(atomMemories.AtomXPos + i) = AtomicStructure[i].x;
		*(atomMemories.AtomYPos + i) = AtomicStructure[i].y;
		*(atomMemories.AtomZPos + i) = AtomicStructure[i].z;
	}

	//Malloc Device Memory
	cudaMalloc(&atomMemories.devAtomXPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomYPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomZPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomZNum,AtomicStructure.size()*sizeof(int));


	// Upload to device :)
	cudaMemcpy(atomMemories.devAtomXPos, atomMemories.AtomXPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomYPos, atomMemories.AtomYPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZPos, atomMemories.AtomZPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);

	// Sort on device
	// Kernel should split itself if too big but it doesnt seem to work :(
	dim3 dimBlock2(512,1,1);
	dim3 dimGrid2(0,0,0);

	if(AtomicStructure.size() > 102400)
	{
		dimGrid2.x = (102400 + dimBlock2.x-1)/dimBlock2.x;
		dimGrid2.y = 1;
		dimGrid2.z = 1;
	}
	else
	{
		dimGrid2.x = (AtomicStructure.size() + dimBlock2.x-1)/dimBlock2.x;
		dimGrid2.y = 1;
		dimGrid2.z = 1;
	}
	
	int runs = ( AtomicStructure.size() + 102399 ) / 102400 ;
	
	// NOTE: DONT CHANGE UNLESS CHANGE ELSEWHERE ASWELL!
	int xBlocks = 50;
	int yBlocks = 50;
	int	dz		= 1;
	int	nSlices	= ceil((MaximumZ-MinimumZ)/dz);
	nSlices+=(nSlices==0);

	//Malloc HBlockStuff
	atomMemories.HBlockIds = new int [AtomicStructure.size()];
	atomMemories.HZIds = new int [AtomicStructure.size()];
	
	cudaMalloc(&atomMemories.DBlockIds,AtomicStructure.size()*sizeof(int));
	cudaMalloc(&atomMemories.DZIds,AtomicStructure.size()*sizeof(int));

	int start = 0;
	for(int i = 1 ; i <= runs; i++ )
	{
		int length = 102400;

		if(i == runs)
		{
			length = AtomicStructure.size() - (runs - 1)*102400;
		}

		atombin2<<<dimBlock2,dimGrid2>>>(atomMemories.devAtomXPos,atomMemories.devAtomYPos,start,length,MaximumX,MinimumX,MaximumY,MinimumY,xBlocks,yBlocks, atomMemories.DBlockIds,atomMemories.DZIds,atomMemories.devAtomZPos,MaximumZ,dz,nSlices);

		start+=102400;
	}

	cudaMemcpy(atomMemories.HBlockIds,atomMemories.DBlockIds,AtomicStructure.size()*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(atomMemories.HZIds,atomMemories.DZIds,AtomicStructure.size()*sizeof(int),cudaMemcpyDeviceToHost);
	
	
	// no longer needed.
	cudaFree(atomMemories.DBlockIds);
	cudaFree(atomMemories.DZIds);

	
	vector < vector < vector < float > > > BinnedX;
	BinnedX.resize(xBlocks*yBlocks);
	vector < vector < vector < float > > > BinnedY;
	BinnedY.resize(xBlocks*yBlocks);
	vector < vector < vector < float > > > BinnedZ;
	BinnedZ.resize(xBlocks*yBlocks);
	vector < vector < vector < int > > > Binnedznum;
	Binnedznum.resize(xBlocks*yBlocks);

	int bigsize = BinnedX.max_size();

	int maxBlockId(0);
	int maxSliceId(0);
	
	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		if(atomMemories.HBlockIds[i] > maxBlockId)
			maxBlockId = atomMemories.HBlockIds[i];
		if(atomMemories.HZIds[i] > maxSliceId)
			maxSliceId = atomMemories.HZIds[i];	
	}


		
	for(int i = 0 ; i < xBlocks*yBlocks ; i++){
		BinnedX[i].resize(nSlices);
		BinnedY[i].resize(nSlices);
		BinnedZ[i].resize(nSlices);
		Binnedznum[i].resize(nSlices);
	}



	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		BinnedX[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomXPos[i]-MinimumX);
		BinnedY[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomYPos[i]-MinimumY);
		BinnedZ[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomZPos[i]-MinimumZ);
		Binnedznum[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomZNum[i]);
	}

	
	int atomIterator(0);
	int* blockStartPositions;
	blockStartPositions = new int[nSlices*xBlocks*yBlocks+1];


	// Put all bins into a linear block of memory ordered by z then y then x and record start positions for every block.
	
	for(int slicei = 0; slicei < nSlices; slicei++)
	{
		for(int j = 0; j < yBlocks; j++)
		{
			for(int k = 0; k < xBlocks; k++)
			{
				blockStartPositions[slicei*xBlocks*yBlocks+ j*xBlocks + k] = atomIterator;

				if(BinnedX[j*xBlocks+k][slicei].size() > 0)
				{
					for(int l = 0; l < BinnedX[j*xBlocks+k][slicei].size(); l++)
					{
						// cout <<"Block " << j <<" , " << k << endl;
						*(atomMemories.AtomXPos+atomIterator) = BinnedX[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomYPos+atomIterator) = BinnedY[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomZPos+atomIterator) = BinnedZ[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomZNum+atomIterator) = Binnedznum[j*xBlocks+k][slicei][l];
						atomIterator++;
					}
				}
			}
		}
	}

	// Last element indicates end of last block as total number of atoms.
	blockStartPositions[nSlices*xBlocks*yBlocks]=AtomicStructure.size();

	cudaMalloc(&atomMemories.devBlockStartPositions,(nSlices*xBlocks*yBlocks+1)*sizeof(int));
	cudaMemcpy(atomMemories.devBlockStartPositions,blockStartPositions,(nSlices*xBlocks*yBlocks+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZNum, atomMemories.AtomZNum, AtomicStructure.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomXPos, atomMemories.AtomXPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomYPos, atomMemories.AtomYPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZPos, atomMemories.AtomZPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);


	// Clear some unnecessary stuff now.
	delete[] blockStartPositions;
	delete[] atomMemories.AtomZNum;
	delete[] atomMemories.AtomXPos;
	delete[] atomMemories.AtomYPos;
	delete[] atomMemories.AtomZPos;
	delete[] atomMemories.HBlockIds;
	delete[] atomMemories.HZIds;
}

void UnmanagedMultisliceSimulation::UploadBinnedAtomsTDS()
{
	// Currently has problems importing some very large structures but not sure which part is causing the problem without proper access to debugging.
	atomMemories.AtomZNum = new int[AtomicStructure.size()];
	atomMemories.AtomXPos = new float[AtomicStructure.size()];
	atomMemories.AtomYPos = new float[AtomicStructure.size()];
	atomMemories.AtomZPos = new float[AtomicStructure.size()];

	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		*(atomMemories.AtomZNum + i) = AtomicStructure[i].atomicNumber;
		*(atomMemories.AtomXPos + i) = AtomicStructure[i].x + TDSRand();
		*(atomMemories.AtomYPos + i) = AtomicStructure[i].y + TDSRand();
		*(atomMemories.AtomZPos + i) = AtomicStructure[i].z + TDSRand();
	}

	//Malloc Device Memory
	cudaMalloc(&atomMemories.devAtomXPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomYPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomZPos,AtomicStructure.size()*sizeof(float));
	cudaMalloc(&atomMemories.devAtomZNum,AtomicStructure.size()*sizeof(int));


	// Upload to device :)
	cudaMemcpy(atomMemories.devAtomXPos, atomMemories.AtomXPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomYPos, atomMemories.AtomYPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZPos, atomMemories.AtomZPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);

	// Sort on device
	// Kernel should split itself if too big but it doesnt seem to work :(
	dim3 dimBlock2(512,1,1);
	dim3 dimGrid2(0,0,0);

	if(AtomicStructure.size() > 102400)
	{
		dimGrid2.x = (102400 + dimBlock2.x-1)/dimBlock2.x;
		dimGrid2.y = 1;
		dimGrid2.z = 1;
	}
	else
	{
		dimGrid2.x = (AtomicStructure.size() + dimBlock2.x-1)/dimBlock2.x;
		dimGrid2.y = 1;
		dimGrid2.z = 1;
	}
	
	int runs = ( AtomicStructure.size() + 102399 ) / 102400 ;
	
	// NOTE: DONT CHANGE UNLESS CHANGE ELSEWHERE ASWELL!
	int xBlocks = 50;
	int yBlocks = 50;
	int	dz		= 1;
	int	nSlices	= ceil((MaximumZ-MinimumZ)/dz);
	nSlices+=(nSlices==0);

	//Malloc HBlockStuff
	atomMemories.HBlockIds = new int [AtomicStructure.size()];
	atomMemories.HZIds = new int [AtomicStructure.size()];
	
	cudaMalloc(&atomMemories.DBlockIds,AtomicStructure.size()*sizeof(int));
	cudaMalloc(&atomMemories.DZIds,AtomicStructure.size()*sizeof(int));

	int start = 0;
	for(int i = 1 ; i <= runs; i++ )
	{
		int length = 102400;

		if(i == runs)
		{
			length = AtomicStructure.size() - (runs - 1)*102400;
		}

		atombin2<<<dimBlock2,dimGrid2>>>(atomMemories.devAtomXPos,atomMemories.devAtomYPos,start,length,MaximumX,MinimumX,MaximumY,MinimumY,xBlocks,yBlocks, atomMemories.DBlockIds,atomMemories.DZIds,atomMemories.devAtomZPos,MaximumZ,dz,nSlices);

		start+=102400;
	}

	cudaMemcpy(atomMemories.HBlockIds,atomMemories.DBlockIds,AtomicStructure.size()*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(atomMemories.HZIds,atomMemories.DZIds,AtomicStructure.size()*sizeof(int),cudaMemcpyDeviceToHost);
	
	
	// no longer needed.
	cudaFree(atomMemories.DBlockIds);
	cudaFree(atomMemories.DZIds);

	
	vector < vector < vector < float > > > BinnedX;
	BinnedX.resize(xBlocks*yBlocks);
	vector < vector < vector < float > > > BinnedY;
	BinnedY.resize(xBlocks*yBlocks);
	vector < vector < vector < float > > > BinnedZ;
	BinnedZ.resize(xBlocks*yBlocks);
	vector < vector < vector < int > > > Binnedznum;
	Binnedznum.resize(xBlocks*yBlocks);

	int bigsize = BinnedX.max_size();

	int maxBlockId(0);
	int maxSliceId(0);
	
	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		if(atomMemories.HBlockIds[i] > maxBlockId)
			maxBlockId = atomMemories.HBlockIds[i];
		if(atomMemories.HZIds[i] > maxSliceId)
			maxSliceId = atomMemories.HZIds[i];	
	}


		
	for(int i = 0 ; i < xBlocks*yBlocks ; i++){
		BinnedX[i].resize(nSlices);
		BinnedY[i].resize(nSlices);
		BinnedZ[i].resize(nSlices);
		Binnedznum[i].resize(nSlices);
	}



	for(int i = 0; i < AtomicStructure.size(); i++)
	{
		BinnedX[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomXPos[i]-MinimumX);
		BinnedY[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomYPos[i]-MinimumY);
		BinnedZ[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomZPos[i]-MinimumZ);
		Binnedznum[atomMemories.HBlockIds[i]][atomMemories.HZIds[i]].push_back(atomMemories.AtomZNum[i]);
	}

	
	int atomIterator(0);
	int* blockStartPositions;
	blockStartPositions = new int[nSlices*xBlocks*yBlocks+1];


	// Put all bins into a linear block of memory ordered by z then y then x and record start positions for every block.
	
	for(int slicei = 0; slicei < nSlices; slicei++)
	{
		for(int j = 0; j < yBlocks; j++)
		{
			for(int k = 0; k < xBlocks; k++)
			{
				blockStartPositions[slicei*xBlocks*yBlocks+ j*xBlocks + k] = atomIterator;

				if(BinnedX[j*xBlocks+k][slicei].size() > 0)
				{
					for(int l = 0; l < BinnedX[j*xBlocks+k][slicei].size(); l++)
					{
						// cout <<"Block " << j <<" , " << k << endl;
						*(atomMemories.AtomXPos+atomIterator) = BinnedX[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomYPos+atomIterator) = BinnedY[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomZPos+atomIterator) = BinnedZ[j*xBlocks+k][slicei][l];
						*(atomMemories.AtomZNum+atomIterator) = Binnedznum[j*xBlocks+k][slicei][l];
						atomIterator++;
					}
				}
			}
		}
	}

	// Last element indicates end of last block as total number of atoms.
	blockStartPositions[nSlices*xBlocks*yBlocks]=AtomicStructure.size();

	cudaMalloc(&atomMemories.devBlockStartPositions,(nSlices*xBlocks*yBlocks+1)*sizeof(int));
	cudaMemcpy(atomMemories.devBlockStartPositions,blockStartPositions,(nSlices*xBlocks*yBlocks+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZNum, atomMemories.AtomZNum, AtomicStructure.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomXPos, atomMemories.AtomXPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomYPos, atomMemories.AtomYPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomMemories.devAtomZPos, atomMemories.AtomZPos, AtomicStructure.size()*sizeof(float), cudaMemcpyHostToDevice);


	// Clear some unnecessary stuff now.
	delete[] blockStartPositions;
	delete[] atomMemories.AtomZNum;
	delete[] atomMemories.AtomXPos;
	delete[] atomMemories.AtomYPos;
	delete[] atomMemories.AtomZPos;
	delete[] atomMemories.HBlockIds;
	delete[] atomMemories.HZIds;
}

void UnmanagedMultisliceSimulation::GetParameterisation()
{
	char inputparamsFilename[] = "fparams.dat";

	// Read in fparams data for calculating projected atomic potential.

	ifstream inparams;
	inparams.open(inputparamsFilename , ios::in);
	
	vector<AtomParameterisation> fparams;
	AtomParameterisation buffer;

	if (!inparams) {
		exit(1);
	}
	
	
	while ((inparams >> buffer.a >> buffer.b >> buffer.c >> buffer.d >> buffer.e >> buffer.f >> buffer.g >> buffer.h >> buffer.i >> buffer.j >> buffer.k >> buffer.l))
	{
	fparams.push_back (buffer);
	}

	inparams.close();

	cudaMalloc(&atomMemories.fparamsdev,12*103*sizeof(float));
	cudaMemcpy(atomMemories.fparamsdev, &fparams[0], 12*103*sizeof(float), cudaMemcpyHostToDevice);

	fparams.clear();
	
}

void UnmanagedMultisliceSimulation::ApplyMicroscopeParameters(float Voltage, float defocus, float Mod2f, float Arg2f, float spherical, float B, float D, float obj)
{
	kV = Voltage;
	df = defocus;
	Mod2fold = Mod2f;
	Arg2fold = Arg2f;
	Cs = spherical;
	beta = B;
	delta = D;
	objectiveAperture = obj;

	
}

void UnmanagedMultisliceSimulation::SetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float sizeX, float sizeY, float sizeZ, float blockoffsetx, float blockoffsety)
{
	// Set class level variables
	resolutionX = resX;
	resolutionY = resY;
	df = defocus;
	
	int SampleinX = resolutionX;
	int SampleinY = resolutionY;

	blockxoffset = blockoffsetx;
	blockyoffset = blockoffsety;


	// Currently this part is deciding size of image to simulate irrespective of the size of the structure, it does however assume the blocks are all based around the origin, so image is always top left?

	PixelScale = PixelScaleIn;
	//PixelScale	= ((sizeY>sizeX)*sizeY+(sizeX>=sizeY)*sizeX)/SampleinY;
	ksizex = PixelScale * SampleinX; // IMPORTANT - if forcing square image need to use image size not structure size for calculating frequencies.
	ksizey = PixelScale * SampleinY;

	// TODO: Link with other location that uses xBlocks.
	int xBlocks = 50;
	int yBlocks = 50;

	float	BlockScaleX = sizeX/xBlocks;
	float	BlockScaleY = sizeY/yBlocks;

	float	Pi		= 3.1415926f;	
	float	V		= kV;
	float	a0		= 52.9177e-012f;
	float	a0a		= a0*1e+010f;
	float	echarge	= 1.6e-019f;
	wavel	= 6.63e-034f*3e+008f/sqrt((echarge*V*1000*(2*9.11e-031f*9e+016f + echarge*V*1000)))*1e+010f;
	float	sigma	= 2 * Pi * ((511 + V) / (2*511 + V)) / (V * wavel);
	sigma2	= (2*Pi/(wavel * V * 1000)) * ((9.11e-031f*9e+016f + echarge*V*1000)/(2*9.11e-031f*9e+016f + echarge*V*1000));
	float	fix		= 300.8242834f/(4*Pi*Pi*a0a*echarge);
	float	V2		= V*1000;


	float fnkx = resolutionX;
	float fnky = resolutionY;

	float p1 = fnkx/(2*ksizex);
	float p2 = fnky/(2*ksizey);
	float p12 = p1*p1;
	float p22 = p2*p2;

	//TODO: Is this even necessary without OpenGl stuff? Probably Not!
	//image_width=SampleinX; // Update so we know how big to make texture after prelimsetup() is done
	//image_height=SampleinY; // Update so we know how big to make texture after prelimsetup() is done

	ke2 = (.666666f)*(p12+p22);

	float quadraticA =(ke2*ke2*16*Pi*Pi*Pi*Pi) - (32*Pi*Pi*Pi*ke2*sigma2*V2/wavel) + (16*Pi*Pi*sigma2*sigma2*V2*V2/(wavel*wavel));
	float quadraticB =16*Pi*Pi*(ke2 - (sigma2*V2/(Pi*wavel)) - (1/(4*wavel*wavel)));
	float quadraticC =3;
	float quadraticB24AC = quadraticB * quadraticB - 4*quadraticA*quadraticC;
	
	// Now use these to determine acceptable resolution or enforce extra band limiting beyond 2/3
	if(quadraticB24AC<0)
	{
		//TODO: Need an actual expection and message for these circumstances..
		/*
		cout << "No stable solution exists for these conditions in FD Multislice" << endl;
		return;
		*/
	}

	float b24ac = sqrtf(quadraticB24AC);
	float maxStableDz = (-quadraticB+b24ac)/(2*quadraticA);
	maxStableDz = 0.99*sqrtf(maxStableDz);

	// Not sure why i wrote this?
	//if(maxStableDz>0.06)
	//	maxStableDz=0.06;


	int	nFDSlices	= ceil((sizeZ)/maxStableDz);
	// Prevent 0 slices for perfectly flat sample
	nFDSlices+=(nFDSlices==0);

	numberOfSlices = nFDSlices;
	dzFiniteDifference = maxStableDz;

	// For atom slicing
	float dz = 1;

	// TODO: Maybe set it in CUDA const cache here aswell, could do with maybe setting up memory somewhere else to allow rewriting, i.e in Constructor;

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	int loadblocksx = ceil(sqrtf(5.0f)/((MaximumX-MinimumX)/(xBlocks)));
	int loadblocksy = ceil(sqrtf(5.0f)/((MaximumY-MinimumY)/(yBlocks)));
	int loadslicesz = ceil(sqrtf(5.0f)/dz);

	int hostconsts[3] = {loadblocksx,loadblocksy,loadslicesz};
	int hostres[2] = {resolutionX,resolutionY};
	float hostmaxs[6] = {MaximumX,MinimumX,MaximumY,MinimumY,MaximumZ,MinimumZ};
	int hostblocks[2] = {dimBlock.x,dimBlock.y};
	int hostgrid[2] = {dimGrid.x,dimGrid.y};
	int hostbins[2] = {xBlocks,yBlocks};

	cudaMemcpyToSymbol("consts",hostconsts,3*sizeof(int),0);
	cudaMemcpyToSymbol("res",hostres,2*sizeof(int),0);
	cudaMemcpyToSymbol("maxs",hostmaxs,6*sizeof(float),0);
	cudaMemcpyToSymbol("blocks",hostblocks,2*sizeof(int),0);
	cudaMemcpyToSymbol("grids",hostgrid,2*sizeof(int),0);
	cudaMemcpyToSymbol("bins",hostbins,2*sizeof(int),0);

}

void UnmanagedMultisliceSimulation::PreCalculateFrequencies()
{
	// Create vectors for pixel frequencies in fourier space.

	int imidx = floor(resolutionX/2 + 0.5);
	int imidy = floor(resolutionY/2 + 0.5);

	std::vector<float> k0x;
	std::vector<float> k0y;

	float temp;

	for(int i=1 ; i <= resolutionX ; i++)
	{
		if ((i - 1) > imidx)
			temp = ((i - 1) - resolutionX)/ksizex;
		else temp = (i - 1)/ksizex;
		k0x.push_back (temp);
	}

	for(int i=1 ; i <= resolutionY ; i++)
	{
		if ((i - 1) > imidy)
			temp = ((i - 1) - resolutionY)/ksizey;
		else temp = (i - 1)/ksizey;
		k0y.push_back (temp);
	}

	int imidxbig = floor(resolutionX + 0.5);
	int imidybig = floor(resolutionY + 0.5);

	std::vector<float> k0xbig;
	std::vector<float> k0ybig;

	for(int i=1 ; i <= 2*resolutionX ; i++)
	{
		if ((i - 1) > imidxbig)
			temp = ((i - 1) - 2*resolutionX)/(2*ksizex);
		else temp = (i - 1)/(2*ksizex);
		k0xbig.push_back (temp);
	}

	for(int i=1 ; i <= 2*resolutionY ; i++)
	{
		if ((i - 1) > imidybig)
			temp = ((i - 1) - 2*resolutionY)/(2*ksizey);
		else temp = (i - 1)/(2*ksizey);
		k0ybig.push_back (temp);
	}

	// Find maximum frequency for bandwidth limiting rule....

	kmax=0;

	float	kmaxx = pow((k0x[imidx-1]*1/2),2);
	float	kmaxy = pow((k0y[imidy-1]*1/2),2);
	
	if(kmaxy <= kmaxx)
	{
		kmax = kmaxy;
	}
	else 
	{ 
		kmax = kmaxx;
	};


	
	// Bandlimit by FDdz size

	// TODO: make sure that this gets followed through when kmax is next used.
	// NOTE: No idea what i meant anymore? think this only matters for doing finite difference method.
	//kmax=sqrtf(ke2);

	// Looks like kmax is a k^2 value.
	// This didnt appear to affect much at all... :(
	kmax = sqrt( kmax );

	// Now upload to device and clear host bits.
	cudaMalloc(&xFrequencies,resolutionX*sizeof(float));
	cudaMalloc(&yFrequencies,resolutionY*sizeof(float));
	cudaMemcpy(xFrequencies, &k0x[0], resolutionX*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yFrequencies, &k0y[0], resolutionY*sizeof(float), cudaMemcpyHostToDevice);

	cufftPlan2d(&plan,resolutionY,resolutionX,CUFFT_C2C);
	normalisingfactor = 1/sqrtf(resolutionX*resolutionY);

}

void UnmanagedMultisliceSimulation::InitialiseWavefunctions()
{
	conventionaldz = 1;

	//TODO: If ever change change everywhere...
	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	cudaMalloc(&PsiMinus,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&Psi,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&PsiPlus,resolutionX*resolutionY*sizeof(cuComplex));

	cuComplex* devPropagationKernel;
	cudaMalloc(&devPropagationKernel,resolutionX*resolutionY*sizeof(cuComplex));

	initializingKernel<<<dimGrid,dimBlock>>>(PsiMinus,1.0f);

	// Launch Kernel to create propogation matrix
	// This is wrong if it is not using Finite Difference.......
	CreatePropsKernel<<<dimGrid,dimBlock>>>(xFrequencies,yFrequencies,dzFiniteDifference,wavel,devPropagationKernel,kmax);

	// Propagate to create second initial wavefunction
	// temporarily borrow other memory for use during FFT
	cufftExecC2C(plan,PsiMinus,Psi,CUFFT_FORWARD);
	multiplicationKernel<<<dimGrid,dimBlock>>>(Psi,devPropagationKernel,PsiPlus,normalisingfactor);
	cufftExecC2C(plan,PsiPlus,Psi,CUFFT_INVERSE);
	normalisingKernel<<<dimGrid,dimBlock>>>(Psi,normalisingfactor);

	cudaFree(devPropagationKernel);

	cudaMalloc(&devV,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad2,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devBandLimitStorage,resolutionX*resolutionY*sizeof(cuComplex));
}

void UnmanagedMultisliceSimulation::InitialiseWavefunctions(float dz)
{
	conventionaldz = dz;

	//TODO: If ever change change everywhere...
	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	cudaMalloc(&PsiMinus,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&Psi,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&PsiPlus,resolutionX*resolutionY*sizeof(cuComplex));

	cuComplex* devPropagationKernel;
	cudaMalloc(&devPropagationKernel,resolutionX*resolutionY*sizeof(cuComplex));

	initializingKernel<<<dimGrid,dimBlock>>>(PsiMinus,1.0f);

	// Launch Kernel to create propogation matrix
	// This is wrong if it is not using Finite Difference.......
	CreatePropsKernel<<<dimGrid,dimBlock>>>(xFrequencies,yFrequencies,dzFiniteDifference,wavel,devPropagationKernel,kmax);

	// Propagate to create second initial wavefunction
	// temporarily borrow other memory for use during FFT
	cufftExecC2C(plan,PsiMinus,Psi,CUFFT_FORWARD);
	multiplicationKernel<<<dimGrid,dimBlock>>>(Psi,devPropagationKernel,PsiPlus,normalisingfactor);
	cufftExecC2C(plan,PsiPlus,Psi,CUFFT_INVERSE);
	normalisingKernel<<<dimGrid,dimBlock>>>(Psi,normalisingfactor);

	cudaFree(devPropagationKernel);

	cudaMalloc(&devV,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad2,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devBandLimitStorage,resolutionX*resolutionY*sizeof(cuComplex));
}

void UnmanagedMultisliceSimulation::MultisliceStep(int iteration)
{
	
		float dz = 1;

		//TODO: If ever change change everywhere...
		dim3 dimBlock(32,8,1);
		dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
		dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	
		float currentz = MaximumZ - MinimumZ - dzFiniteDifference*iteration;
			
		binnedAtomPKernelFD<<<dimGrid,dimBlock>>>(devV,PixelScale,atomMemories.fparamsdev,atomMemories.devAtomZPos,atomMemories.devAtomXPos,atomMemories.devAtomYPos,atomMemories.devAtomZNum,atomMemories.devBlockStartPositions,dz,currentz,ceil((MaximumZ-MinimumZ)/dz));
		
		
		//cudaMemcpy(checkval,&devV[40000],1*sizeof(cuComplex),cudaMemcpyDeviceToHost);
	//	cout << checkval[0].x <<" , " << checkval[0].y << endl;

		
		cufftExecC2C(plan,devV,devBandLimitStorage,CUFFT_FORWARD);
		BandLimitKernel<<<dimGrid,dimBlock>>>(devBandLimitStorage,kmax,xFrequencies,yFrequencies);
		cufftExecC2C(plan,devBandLimitStorage,devV,CUFFT_INVERSE);
		normalisingKernel<<<dimGrid,dimBlock>>>(devV,normalisingfactor*normalisingfactor);
		
	
		cufftExecC2C(plan,Psi,devGrad2,CUFFT_FORWARD);
		gradKernel<<<dimGrid,dimBlock>>>(devGrad2,xFrequencies,yFrequencies,normalisingfactor);
		cufftExecC2C(plan,devGrad2,devGrad,CUFFT_INVERSE);
		normalisingKernel<<<dimGrid,dimBlock>>>(devGrad,normalisingfactor);



		finiteDifferenceKernel<<<dimGrid,dimBlock>>>(devGrad,PsiMinus,PsiPlus,Psi,devV,sigma2,wavel,dzFiniteDifference);

		cufftExecC2C(plan,PsiPlus,devBandLimitStorage,CUFFT_FORWARD);
		BandLimitKernel<<<dimGrid,dimBlock>>>(devBandLimitStorage,kmax,xFrequencies,yFrequencies);
		cufftExecC2C(plan,devBandLimitStorage,PsiPlus,CUFFT_INVERSE);
		normalisingKernel<<<dimGrid,dimBlock>>>(PsiPlus,normalisingfactor*normalisingfactor);


		cudaMemcpy(PsiMinus,Psi,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);
		cudaMemcpy(Psi,PsiPlus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);
}

void UnmanagedMultisliceSimulation::MultisliceStepConv(int iteration)
{
	
		float dz = 1;

		//TODO: If ever change change everywhere...
		dim3 dimBlock(32,8,1);
		dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
		dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

		//Set Conventional dz = 1 for really old method..
	
		float currentz = MaximumZ - MinimumZ - conventionaldz*iteration;
			
		// NOTE getting some NaN's in Psi.

		binnedAtomPKernel3<<<dimGrid,dimBlock>>>(devV,PixelScale,atomMemories.fparamsdev,atomMemories.devAtomZPos,atomMemories.devAtomXPos,atomMemories.devAtomYPos,atomMemories.devAtomZNum,atomMemories.devBlockStartPositions,dz,conventionaldz,currentz,ceil((MaximumZ-MinimumZ)/dz),sigma2,blockxoffset,blockyoffset);
		
		if(iteration==1)
		{
			multiplicationKernel<<<dimGrid,dimBlock>>>(devV,PsiMinus,PsiPlus,1); // Think initialized psi with one not psiminus
		}
		else
		{
			multiplicationKernel<<<dimGrid,dimBlock>>>(devV,Psi,PsiPlus,1);
		}

	//  cudaMemcpy(checkval,&devV[40000],1*sizeof(cuComplex),cudaMemcpyDeviceToHost);
	//	cout << checkval[0].x <<" , " << checkval[0].y << endl;
		
		// Now it
		if(iteration==1)
			CreatePropsKernel<<<dimGrid,dimBlock>>>(xFrequencies,yFrequencies,conventionaldz,wavel,PsiMinus,kmax);
		
		cufftExecC2C(plan,PsiPlus,Psi,CUFFT_FORWARD);
		multiplicationKernel<<<dimGrid,dimBlock>>>(Psi,PsiMinus,PsiPlus,normalisingfactor);
		cufftExecC2C(plan,PsiPlus,Psi,CUFFT_INVERSE);
		normalisingKernel<<<dimGrid,dimBlock>>>(Psi,normalisingfactor);
		
}

void UnmanagedMultisliceSimulation::MultisliceStepRS(int iteration)
{
	
		float dz = 1;

		//TODO: If ever change change everywhere...
		dim3 dimBlock(32,8,1);
		dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
		dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	
		float currentz = MaximumZ - MinimumZ - dz*iteration;


		binnedAtomPKernelRS<<<dimGrid,dimBlock>>>(devV,PixelScale,atomMemories.fparamsdev,atomMemories.devAtomZPos,atomMemories.devAtomXPos,atomMemories.devAtomYPos,atomMemories.devAtomZNum,atomMemories.devBlockStartPositions,dz,currentz,ceil((MaximumZ-MinimumZ)/dz),sigma2,blockxoffset,blockyoffset);
		
		if(iteration==1)
		{
			for(int i = 1 ; i <=10 ; i++)
			{
				DifferentialXKernel<<<dimGrid,dimBlock>>>(PsiPlus,devGrad,PixelScale);
				DifferentialYKernel<<<dimGrid,dimBlock>>>(PsiPlus,devGrad2,PixelScale);

				RSMS<<<dimGrid,dimBlock>>>(devGrad,devGrad2,PsiPlus,devV,PsiMinus,wavel,dz,i);	
			}

				multiplicationKernel<<<dimGrid,dimBlock>>>(PsiPlus,PsiMinus,Psi,1.0f);

			//cudaMemcpy(PsiMinus,PsiPlus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);
		}
		else
		{
			for(int i = 1 ; i <=10 ; i++)
			{
				DifferentialXKernel<<<dimGrid,dimBlock>>>(PsiPlus,devGrad,PixelScale);
				DifferentialYKernel<<<dimGrid,dimBlock>>>(PsiPlus,devGrad2,PixelScale);

				RSMS<<<dimGrid,dimBlock>>>(devGrad,devGrad2,PsiPlus,devV,PsiMinus,wavel,dz,i);
			}

			multiplicationKernel<<<dimGrid,dimBlock>>>(PsiPlus,Psi,PsiMinus,1.0f);
				//cudaMemcpy(Psi,PsiPlus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);

			// Copy to Psi for next iteration....
		cudaMemcpy(Psi,PsiPlus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);
			
		}

		
		// Copy to Psi for next iteration....
		cudaMemcpy(Psi,PsiPlus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToDevice);
		
}

void UnmanagedMultisliceSimulation::GetSTEMExitWave()
{

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	
	EW = new float2[resolutionX*resolutionY];

	cufftExecC2C(plan,Psi,PsiPlus,CUFFT_FORWARD);

	CentreKernel<<<dimGrid,dimBlock>>>(PsiPlus,PsiMinus);

	cudaMemcpy(EW,PsiMinus,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToHost);


	float* absimage;
	cudaMalloc(&absimage,resolutionX*resolutionY*sizeof(float));
	
	conjugationKernel<<<dimGrid,dimBlock>>>(PsiMinus,absimage,1,0);// Should be 1

	int maxpos;
	int minpos;

	cublasstatus = cublasIsamax(cublashandle,resolutionX*resolutionY,absimage,1,&maxpos); // Find index of maximum value to scale image display
	cublasstatus = cublasIsamin(cublashandle,resolutionX*resolutionY,absimage,1,&minpos); // Find index of maximum value to scale image display


	maxpos-=1;
	minpos-=1;

	cuComplex* maxval = new cuComplex [1];
	cuComplex* minval = new cuComplex [1];

	maxval[0] = EW[maxpos];
	minval[0] = EW[minpos];

	//cudaMemcpy(maxval,&Psi[maxpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);
	//cudaMemcpy(minval,&Psi[minpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);

	maxposEW = sqrt((maxval[0].x * maxval[0].x) + (maxval[0].y * maxval[0].y));
	minposEW = sqrt((minval[0].x * minval[0].x) + (minval[0].y * minval[0].y));


	//maxposEW = 1.5f;

	delete maxval;
	delete minval;
	cudaFree(absimage);
	
}

void UnmanagedMultisliceSimulation::GetExitWave()
{

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	
	EW = new float2[resolutionX*resolutionY];

	cudaMemcpy(EW,Psi,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToHost);


	float* absimage;
	cudaMalloc(&absimage,resolutionX*resolutionY*sizeof(float));
	
	conjugationKernel<<<dimGrid,dimBlock>>>(Psi,absimage,1,0);// Should be 1

	int maxpos;
	int minpos;

	cublasstatus = cublasIsamax(cublashandle,resolutionX*resolutionY,absimage,1,&maxpos); // Find index of maximum value to scale image display
	cublasstatus = cublasIsamin(cublashandle,resolutionX*resolutionY,absimage,1,&minpos); // Find index of maximum value to scale image display


	maxpos-=1;
	minpos-=1;

	cuComplex* maxval = new cuComplex [1];
	cuComplex* minval = new cuComplex [1];

	maxval[0] = EW[maxpos];
	minval[0] = EW[minpos];

	//cudaMemcpy(maxval,&Psi[maxpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);
	//cudaMemcpy(minval,&Psi[minpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);

	maxposEW = sqrt((maxval[0].x * maxval[0].x) + (maxval[0].y * maxval[0].y));
	minposEW = sqrt((minval[0].x * minval[0].x) + (minval[0].y * minval[0].y));


	//maxposEW = 1.5f;

	delete maxval;
	delete minval;
	cudaFree(absimage);
	
		// Clear Up memory now...
	cudaFree(devV);
	cudaFree(devGrad);
	cudaFree(devBandLimitStorage);
	cudaFree(devGrad2);
	//cudaFree(Psi); // Still need
	cudaFree(PsiMinus);
	//cudaFree(PsiPlus); // Still need
}

void UnmanagedMultisliceSimulation::FreeExitWave()
{
	delete[] EW;
}

float UnmanagedMultisliceSimulation::GetEWValueAbs(int xpos, int ypos)
{
	return (sqrt((EW[xpos + ypos*resolutionX].x)*(EW[xpos + ypos*resolutionX].x) + (EW[xpos + ypos*resolutionX].y)*(EW[xpos + ypos*resolutionX].y)));
}

float UnmanagedMultisliceSimulation::GetEWValueRe(int xpos, int ypos)
{
	return (EW[xpos + ypos*resolutionX].x);
}

float UnmanagedMultisliceSimulation::GetEWValueIm(int xpos, int ypos)
{
	return (EW[xpos + ypos*resolutionX].y);
}

void UnmanagedMultisliceSimulation::AllocHostImage()
{
	SimulatedImage = new float2[resolutionX*resolutionY];
}

void UnmanagedMultisliceSimulation::SimulateImage(float doseperpix)
{
	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);

	cuComplex* devImage;
	cudaMalloc(&devImage,resolutionX*resolutionY*sizeof(cuComplex));

	cuComplex* devImage2;
	cudaMalloc(&devImage2,resolutionX*resolutionY*sizeof(cuComplex));
	
	// Borrow memory used for PsiPlus for the FFT of Psi for imaging calculation
	cufftExecC2C(plan,Psi,PsiPlus,CUFFT_FORWARD);

	normalisingKernel<<<dimGrid,dimBlock>>>(PsiPlus,normalisingfactor);

	imagingKernel2<<<dimGrid,dimBlock>>>(PsiPlus,resolutionX,resolutionY,Cs,df,Mod2fold,Arg2fold,Mod3fold,Arg3fold,objectiveAperture,wavel,devImage2,xFrequencies,yFrequencies,beta/1000,delta);

	cufftExecC2C(plan,devImage2,devImage,CUFFT_INVERSE);

	// devImage2 is FFT of image intensity....
	// want to get modules and fft it not just take fft.
	//Multiply this by DQE taking account of scaling and binning etc... :(
	normalisingKernel<<<dimGrid,dimBlock>>>(devImage,normalisingfactor);

	//cudaMemcpy(SimulatedImage,devImage,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToHost);

	// Now apply processing to generate a more realistic image with noise included....
	float* absimage;
	cudaMalloc(&absimage,resolutionX*resolutionY*sizeof(float));

	conjugationKernel<<<dimGrid,dimBlock>>>(devImage,absimage,1,0);
	makeComplexSq<<<dimGrid,dimBlock>>>(absimage,devImage);

	cufftExecC2C(plan,devImage,devImage2,CUFFT_FORWARD);
	normalisingKernel<<<dimGrid,dimBlock>>>(devImage2,normalisingfactor);

	float conversionfactor = 8; //CCD counts per electron.
	float Ntot = doseperpix; // Get this passed in, its dose per pixel i think.

	//load DQE and NTF into cuda memories...
	float* dqemem;
	cudaMalloc(&dqemem,725*sizeof(float));
	float* ntfmem;
	cudaMalloc(&ntfmem,725*sizeof(float));

	cudaMemcpy(ntfmem,k2NTF,725*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dqemem,k2DQE,725*sizeof(float),cudaMemcpyHostToDevice);

	DQEKernel<<<dimGrid,dimBlock>>>(devImage2,dqemem,resolutionX,resolutionY,devImage,4);

	// Inverse and normalise
	cufftExecC2C(plan,devImage,devImage,CUFFT_INVERSE);
	normalisingKernel<<<dimGrid,dimBlock>>>(devImage,normalisingfactor);

	conjugationKernel<<<dimGrid,dimBlock>>>(devImage,absimage,1,0);


	// double stddev = Math.Sqrt(Dose * PixelScale * PixelScale) / (Dose * PixelScale * PixelScale);
    //    Random rand = new Random(); //reuse this if you are generating many
    //    double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
    //    double u2 = rand.NextDouble();
    //    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
    //    Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
    //    double randNormal =
    //    stddev * randStdNormal; //random normal(mean,stdDev^2)

	// Multiply by electron dose, get random from distribution , multiply by conversion factor...

	std::vector<float> hosti1(resolutionX*resolutionY);
	cudaMemcpy(&hosti1[0],absimage,resolutionY*resolutionX*sizeof(float),cudaMemcpyDeviceToHost);
	srand (time(NULL));

	
	//boost::mt19937 randgen(std::time(NULL));

	for(int i = 0; i < resolutionX*resolutionY; i++)
	{
		double random = ((double) rand() / (RAND_MAX+1));
		double random2 = ((double) rand() / (RAND_MAX+1));
		double rstdnormal = sqrt(-2.0f * +log(FLT_MIN+random))*(sin(2.0f * 3.1415926f * random2));
		
		
	if(rstdnormal <= -3)
		rstdnormal = -3;

	if(rstdnormal >= 3)
		rstdnormal = 3;

	//	boost::poisson_distribution<> pd(Ntot*hosti1[i]);
		
	//	boost::variate_generator <boost::mt19937, boost::poisson_distribution<> >generator(randgen,pd);
	//	int value = generator();
		hosti1[i] = (round(Ntot*hosti1[i] + sqrt(fabs(Ntot*hosti1[i]))*rstdnormal));
		//hosti1[i] = (value);


	}

	// Fourier and multiply by NTF... Add random readout noise of correct amplitude.... (Ignore dark current its small....)
	// Did this in final image readout stage instead...


	cudaMemcpy(absimage,&hosti1[0],resolutionY*resolutionX*sizeof(float),cudaMemcpyHostToDevice);
	makeComplex<<<dimGrid,dimBlock>>>(absimage,devImage);
	cufftExecC2C(plan,devImage,devImage,CUFFT_FORWARD);
	normalisingKernel<<<dimGrid,dimBlock>>>(devImage,normalisingfactor);




	NTFKernel<<<dimGrid,dimBlock>>>(devImage,ntfmem,resolutionX,resolutionY,devImage2,4);
	cufftExecC2C(plan,devImage2,devImage,CUFFT_INVERSE);
	normalisingKernel<<<dimGrid,dimBlock>>>(devImage,conversionfactor*conversionfactor*normalisingfactor);

	conjugationKernel<<<dimGrid,dimBlock>>>(devImage,absimage,1,0);

	cudaMemcpy(SimulatedImage,devImage,resolutionX*resolutionY*sizeof(cuComplex),cudaMemcpyDeviceToHost);

	int maxpos;
	int minpos;

	cublasstatus = cublasIsamax(cublashandle,resolutionX*resolutionY,absimage,1,&maxpos); // Find index of maximum value to scale image display
	cublasstatus = cublasIsamin(cublashandle,resolutionX*resolutionY,absimage,1,&minpos); // Find index of maximum value to scale image display

	maxpos-=1;
	minpos -=1;

	cuComplex* maxval = new cuComplex [1];
	cuComplex* minval = new cuComplex [1];

	cudaMemcpy(maxval,&devImage[maxpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(minval,&devImage[minpos],sizeof(cuComplex),cudaMemcpyDeviceToHost);

	maxposIm = sqrt((maxval[0].x * maxval[0].x) + (maxval[0].y * maxval[0].y));
	minposIm = sqrt((minval[0].x * minval[0].x) + (minval[0].y * minval[0].y));

	delete maxval;
	delete minval;
	cudaFree(devImage);


		// Step 1 : Setup smaller memory chunk for subregion of image and copy data in.

	int t(0);
	int l(0);
	int b(512);
	int r(512);

	int subwidth = r - l;
	int subheight = b - t;
	int subarea = subwidth * subheight;

	float * subareaimage;
	cudaMalloc(&subareaimage,subarea*sizeof(float));

	dim3 dimBlock2(32,8,1);
	dim3 dimGrid2((subwidth + dimBlock.x-1)/dimBlock.x,(subheight + dimBlock.y-1)/dimBlock.y,1);

	// For now just top 512,512
	SubImageKernel<<<dimGrid2,dimBlock2>>>(absimage,subareaimage,0,0,512,512,resolutionX,resolutionY);

	// Step 2 : Find average value in region.

	float averageval;
	cublasstatus = cublasSasum(cublashandle,subarea,subareaimage,1,&averageval);

	averageval/=subarea;

	// Step 3 : Calculate (Iij - Iaverage)^2 for all ij

	StdDevKernel<<<dimGrid2,dimBlock2>>>(subareaimage,subwidth,subheight,averageval);

	// Step 4 : Find average again.

	float averageval2;
	cublasstatus = cublasSasum(cublashandle,subarea,subareaimage,1,&averageval2);

	cudaFree(subareaimage);

	imagecontrast = sqrtf(averageval2/subarea);


	cudaFree(absimage);
}

float UnmanagedMultisliceSimulation::GetImValue(int xpos, int ypos)
{
	// Add some readout noise...
	double random = ((double) rand() / (RAND_MAX+1));
	double random2 = ((double) rand() / (RAND_MAX+1));
	double rstdnormal = sqrt(-2.0f * +log(FLT_MIN+random))*(sin(2.0f * 3.1415926f * random2));

	if(rstdnormal <= -3)
		rstdnormal = -3;

	if(rstdnormal >= 3)
		rstdnormal = 3;


	return (4+rstdnormal*1.33+sqrt((SimulatedImage[xpos + ypos*resolutionX].x)*(SimulatedImage[xpos + ypos*resolutionX].x) + (SimulatedImage[xpos + ypos*resolutionX].y)*(SimulatedImage[xpos + ypos*resolutionX].y)));
}

float UnmanagedMultisliceSimulation::GetDQEImValue(int xpos, int ypos)
{
	return (sqrt((SimulatedImage[xpos + ypos*resolutionX].x)*(SimulatedImage[xpos + ypos*resolutionX].x) + (SimulatedImage[xpos + ypos*resolutionX].y)*(SimulatedImage[xpos + ypos*resolutionX].y)));
}

/* Just rolled into end of simulation 

float UnmanagedMultisliceSimulation::GetImageContrast(int t, int l, int b, int r)
{
	// Step 1 : Setup smaller memory chunk for subregion of image and copy data in.

	int subwidth = r - l;
	int subheight = b - t;
	int subarea = subwidth * subheight;

	float * subareaimage;
	cudaMalloc(&subareaimage,subarea*sizeof(float));

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((subwidth + dimBlock.x-1)/dimBlock.x,(subheight + dimBlock.y-1)/dimBlock.y,1);


	SubImageKernel<<<dimBlock,dimGrid>>>();

	// Step 2 : Find average value in region.

	// Step 3 : Calculate (Iij - Iaverage)^2 for all ij

	// Step 4 : Find average again.


	cudaFree(subareaimage);
}
*/



UnmanagedMultisliceSimulation::UnmanagedMultisliceSimulation()
{
	cublasstatus = cublasCreate(&cublashandle);
	Mod3fold = 0;
	Arg3fold = 0;
}

UnmanagedMultisliceSimulation::~UnmanagedMultisliceSimulation()
{
	cublasDestroy(cublashandle);
}



// STEM only functions

void UnmanagedMultisliceSimulation::STEMSetCalculationVariables(float PixelScaleIn, float defocus, int resX, int resY, float sizeX, float sizeY, float sizeZ)
{
	// Set class level variables
	resolutionX = resX;
	resolutionY = resY;
	df = defocus;
	
	int SampleinX = resolutionX;
	int SampleinY = resolutionY;
	//PixelScale	= ((sizeY>sizeX)*sizeY+(sizeX>=sizeY)*sizeX)/SampleinY;
	PixelScale = PixelScaleIn;

	// Not sure about using these frequencies and pixelscale?


	ksizex = PixelScale * SampleinX; // IMPORTANT - if forcing square image need to use image size not structure size for calculating frequencies.
	ksizey = PixelScale * SampleinY;

	// TODO: Link with other location that uses xBlocks.
	int xBlocks = 50;
	int yBlocks = 50;

	float	BlockScaleX = sizeX/xBlocks;
	float	BlockScaleY = sizeY/yBlocks;

	float	Pi		= 3.1415926f;	
	float	V		= kV;
	float	a0		= 52.9177e-012f;
	float	a0a		= a0*1e+010f;
	float	echarge	= 1.6e-019f;
	wavel	= 6.63e-034f*3e+008f/sqrt((echarge*V*1000*(2*9.11e-031f*9e+016f + echarge*V*1000)))*1e+010f;
	float	sigma	= 2 * Pi * ((511 + V) / (2*511 + V)) / (V * wavel);
	sigma2	= (2*Pi/(wavel * V * 1000)) * ((9.11e-031f*9e+016f + echarge*V*1000)/(2*9.11e-031f*9e+016f + echarge*V*1000));
	float	fix		= 300.8242834f/(4*Pi*Pi*a0a*echarge);
	float	V2		= V*1000;


	float fnkx = resolutionX;
	float fnky = resolutionY;

	float p1 = fnkx/(2*ksizex);
	float p2 = fnky/(2*ksizey);
	float p12 = p1*p1;
	float p22 = p2*p2;

	ke2 = (.666666f)*(p12+p22);

	float quadraticA =(ke2*ke2*16*Pi*Pi*Pi*Pi) - (32*Pi*Pi*Pi*ke2*sigma2*V2/wavel) + (16*Pi*Pi*sigma2*sigma2*V2*V2/(wavel*wavel));
	float quadraticB =16*Pi*Pi*(ke2 - (sigma2*V2/(Pi*wavel)) - (1/(4*wavel*wavel)));
	float quadraticC =3;
	float quadraticB24AC = quadraticB * quadraticB - 4*quadraticA*quadraticC;
	
	// Now use these to determine acceptable resolution or enforce extra band limiting beyond 2/3
	if(quadraticB24AC<0)
	{
		//TODO: Need an actual expection and message for these circumstances..
		/*
		cout << "No stable solution exists for these conditions in FD Multislice" << endl;
		return;
		*/
	}

	float b24ac = sqrtf(quadraticB24AC);
	float maxStableDz = (-quadraticB+b24ac)/(2*quadraticA);
	maxStableDz = 0.99*sqrtf(maxStableDz);

	if(maxStableDz>0.06)
		maxStableDz=0.06;


	int	nFDSlices	= ceil((sizeZ)/maxStableDz);
	// Prevent 0 slices for perfectly flat sample
	nFDSlices+=(nFDSlices==0);

	numberOfSlices = nFDSlices;
	dzFiniteDifference = maxStableDz;

	float dz = 1;

	// TODO: Maybe set it in CUDA const cache here aswell, could do with maybe setting up memory somewhere else to allow rewriting, i.e in Constructor;

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	int loadblocksx = ceil(sqrtf(5.0f)/((MaximumX-MinimumX)/(xBlocks)));
	int loadblocksy = ceil(sqrtf(5.0f)/((MaximumY-MinimumY)/(yBlocks)));
	int loadslicesz = ceil(sqrtf(5.0f)/dz);

	int hostconsts[3] = {loadblocksx,loadblocksy,loadslicesz};
	int hostres[2] = {resolutionX,resolutionY};
	float hostmaxs[6] = {MaximumX,MinimumX,MaximumY,MinimumY,MaximumZ,MinimumZ};
	int hostblocks[2] = {dimBlock.x,dimBlock.y};
	int hostgrid[2] = {dimGrid.x,dimGrid.y};
	int hostbins[2] = {xBlocks,yBlocks};

	cudaMemcpyToSymbol("consts",hostconsts,3*sizeof(int),0);
	cudaMemcpyToSymbol("res",hostres,2*sizeof(int),0);
	cudaMemcpyToSymbol("maxs",hostmaxs,6*sizeof(float),0);
	cudaMemcpyToSymbol("blocks",hostblocks,2*sizeof(int),0);
	cudaMemcpyToSymbol("grids",hostgrid,2*sizeof(int),0);
	cudaMemcpyToSymbol("bins",hostbins,2*sizeof(int),0);

}

void UnmanagedMultisliceSimulation::STEMInitialiseWavefunctions(int posx, int posy)
{

	// Also need all of the aberrations for generating the probe... :(

	//TODO: If ever change change everywhere...
	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	dim3 dimGridbig((2*resolutionX + dimBlock.x-1)/dimBlock.x,(2*resolutionY + dimBlock.y-1)/dimBlock.y,1);

	cudaMalloc(&PsiMinus,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&Psi,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&PsiPlus,resolutionX*resolutionY*sizeof(cuComplex));

	cuComplex* devPropagationKernel;
	cudaMalloc(&devPropagationKernel,resolutionX*resolutionY*sizeof(cuComplex));

	//initializingKernel<<<dimGrid,dimBlock>>>(PsiMinus,1.0f);
	STEMinitializingKernel<<<dimGrid,dimBlock>>>(PsiPlus,xFrequencies,yFrequencies,posx,posy,objectiveAperture*0.001/wavel,PixelScale,df,Cs,wavel);

	// Still needs inverseFFTing and normalising.
	cufftExecC2C(plan,PsiPlus,PsiMinus,CUFFT_INVERSE);

	float* absimage;
	cudaMalloc(&absimage,resolutionX*resolutionY*sizeof(float));
	
	conjugationKernel<<<dimGrid,dimBlock>>>(PsiMinus,absimage,1,0);// Should be 1

	// Now get total...

	float normal;
	cublasSasum(cublashandle,resolutionX*resolutionY,absimage,1,&normal);

	normalisingKernel<<<dimGrid,dimBlock>>>(PsiMinus,1/normal);
	

	// Launch Kernel to create propogation matrix
	CreatePropsKernel<<<dimGrid,dimBlock>>>(xFrequencies,yFrequencies,dzFiniteDifference,wavel,devPropagationKernel,kmax);

	// Propagate to create second initial wavefunction
	// temporarily borrow other memory for use during FFT
	cufftExecC2C(plan,PsiMinus,Psi,CUFFT_FORWARD);
	multiplicationKernel<<<dimGrid,dimBlock>>>(Psi,devPropagationKernel,PsiPlus,normalisingfactor);
	cufftExecC2C(plan,PsiPlus,Psi,CUFFT_INVERSE);
	normalisingKernel<<<dimGrid,dimBlock>>>(Psi,normalisingfactor);

	cudaFree(devPropagationKernel);

	cudaMalloc(&devV,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devGrad2,resolutionX*resolutionY*sizeof(cuComplex));
	cudaMalloc(&devBandLimitStorage,resolutionX*resolutionY*sizeof(cuComplex));
}

void UnmanagedMultisliceSimulation::AllocTDS()
{
	//Malloc TDS result
	cudaMallocHost(&TDS,resolutionX*resolutionY*sizeof(float));
	cudaMallocHost(&TDS2,resolutionX*resolutionY*sizeof(float));
	memset(TDS,0,resolutionX*resolutionY*sizeof(float));
}

void UnmanagedMultisliceSimulation::AddTDSWaves()
{
	// EW Currently stored in Psi (Get to Abs value in detector before averaging) 

	dim3 dimBlock(32,8,1);
	dim3 dimGrid((resolutionX + dimBlock.x-1)/dimBlock.x,(resolutionY + dimBlock.y-1)/dimBlock.y,1);
	
	cufftExecC2C(plan,Psi,PsiPlus,CUFFT_FORWARD);

	CentreKernel<<<dimGrid,dimBlock>>>(PsiPlus,PsiMinus);

	float* absimage;
	cudaMalloc(&absimage,resolutionX*resolutionY*sizeof(float));
	
	conjugationKernel<<<dimGrid,dimBlock>>>(PsiMinus,absimage,1,0);// Should be 1

	cudaMemcpy(TDS2,absimage,resolutionX*resolutionY*sizeof(float),cudaMemcpyDeviceToHost);

	for(int j = 0; j < resolutionX*resolutionY; j++)
	{
		TDS[j] += TDS2[j];
	}
	cudaFree(absimage);

	// Clear Up memory now...
	cudaFree(devV);
	cudaFree(devGrad);
	cudaFree(devBandLimitStorage);
	cudaFree(devGrad2);
	cudaFree(Psi);
	cudaFree(PsiMinus);
	cudaFree(PsiPlus);
}

float UnmanagedMultisliceSimulation::GetTDSMin()
{
	float min = FLT_MAX;
	for(int j = 0; j < resolutionX*resolutionY; j++)
	{
		if(TDS[j] < min )
			min = TDS[j];
	}
	return min;
}
float UnmanagedMultisliceSimulation::GetTDSMax()
{
	float max = FLT_MIN;
	for(int j = 0; j < resolutionX*resolutionY; j++)
	{
		if(TDS[j] > max )
			max = TDS[j];
	}
	return max;
}
float UnmanagedMultisliceSimulation::GetTDSVal(int xpos, int ypos)
{
	return TDS[xpos + resolutionX*ypos];
}