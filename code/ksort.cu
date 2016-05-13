#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<time.h>
#include<iostream>

#include "distance.cuh"

// Quicksort partition function
__device__ int partition(double *distances, int *idxs, int left, int right) {
	double pivot = distances[right], dummyDistance;
	int newPivotIdx = left, dummyIdx;
	for (int i = left; i < right; i++) {
		if (distances[i] < pivot) {
			// Swap
			dummyDistance = distances[newPivotIdx];
			distances[newPivotIdx] = distances[i];
			distances[i] = dummyDistance;
			dummyIdx = idxs[newPivotIdx];
			idxs[newPivotIdx] = idxs[i];
			idxs[i] = dummyIdx;
			newPivotIdx++;
		}
	}
	dummyDistance = distances[newPivotIdx];
	distances[newPivotIdx] = distances[right];
	distances[right] = dummyDistance;
	dummyIdx = idxs[newPivotIdx];
	idxs[newPivotIdx] = idxs[right];
	idxs[right] = dummyIdx;

	return newPivotIdx;
}

// Quicksort partition function for item IDs
__device__ int partition(int *idxs, int left, int right) {
	int pivot = idxs[right], dummyIdx, newPivotIdx = left;
	for (int i = left; i < right; i++) {
		if (idxs[i] < pivot) {
			// Swap
			dummyIdx = idxs[newPivotIdx];
			idxs[newPivotIdx] = idxs[i];
			idxs[i] = dummyIdx;
			newPivotIdx++;
		}
	}
	dummyIdx = idxs[newPivotIdx];
	idxs[newPivotIdx] = idxs[right];
	idxs[right] = dummyIdx;

	return newPivotIdx;
}

// Quicksort partition function, descending order
__device__ int rpartition(double *distances, int *idxs, int left, int right) {
	double pivot = distances[right], dummyDistance;
	int newPivotIdx = left, dummyIdx;
	for (int i = left; i < right; i++) {
		if (distances[i] > pivot) {
			// Swap
			dummyDistance = distances[newPivotIdx];
			distances[newPivotIdx] = distances[i];
			distances[i] = dummyDistance;
			dummyIdx = idxs[newPivotIdx];
			idxs[newPivotIdx] = idxs[i];
			idxs[i] = dummyIdx;
			newPivotIdx++;
		}
	}
	dummyDistance = distances[newPivotIdx];
	distances[newPivotIdx] = distances[right];
	distances[right] = dummyDistance;
	dummyIdx = idxs[newPivotIdx];
	idxs[newPivotIdx] = idxs[right];
	idxs[right] = dummyIdx;

	return newPivotIdx;
}

// Quicksort function
__device__ void quicksort(double* distances, int* idxs, int left, int right, bool ascending) {
	
	while (left < right) {
		int newPivotIdx;
		if (ascending)
			newPivotIdx = partition(distances, idxs, left, right);
		else
			newPivotIdx = rpartition(distances, idxs, left, right);
		if (newPivotIdx - left > right - newPivotIdx) {
			quicksort(distances, idxs, newPivotIdx+1, right, ascending);
			right = newPivotIdx - 1;
		}
		else {
			quicksort(distances, idxs, left, newPivotIdx-1, ascending);
			left = newPivotIdx + 1;
		}
	}
}

// Quicksort function, for item IDs
__device__ void quicksort(int* idxs, int left, int right) {
	
	while (left < right) {
		int newPivotIdx = partition(idxs, left, right);
		if (newPivotIdx - left > right - newPivotIdx) {
			quicksort(idxs, newPivotIdx+1, right);
			right = newPivotIdx - 1;
		}
		else {
			quicksort(idxs, left, newPivotIdx-1);
			left = newPivotIdx + 1;
		}
	}
}

// Partial-k sort
__device__ void partialSort(double* distances, int* idxs, int left, int right, int k, bool ascending) {
	while (left < right) {
		int newPivotIdx;
		if (ascending)
			newPivotIdx = partition(distances, idxs, left, right);
		else
			newPivotIdx = rpartition(distances, idxs, left, right);
		if (k < newPivotIdx) {
			right = newPivotIdx - 1;
		} else if (newPivotIdx - left > right - newPivotIdx) {
			quicksort(distances, idxs, newPivotIdx+1, right, ascending);
			right = newPivotIdx - 1;
		} else {
			quicksort(distances, idxs, left, newPivotIdx-1, ascending);
			left = newPivotIdx + 1;
		}
	}
}

// Gets a subset of distances between a given item and a set of rated items.
// Argument "idxs" refers to item IDs.
__device__ void distanceSubset(double* distances, int numItems, int* ratedItems, int numRatedItems, int itemId, double *subset, int *idxs) {
	int i, idx = 0, rIdx = 0, incrementLarge = numItems - 1, incrementSmall = itemId - 1;

	for (i = 0; i < itemId && rIdx < numRatedItems; i++) {
		if (i == ratedItems[rIdx]) {
			subset[rIdx] = distances[idx+incrementSmall];
			idxs[rIdx] = i;
			rIdx++;
		}
		incrementSmall--;
		idx += incrementLarge--;
	}
	for (; i < numItems && rIdx < numRatedItems; i++) {
		if (i+1 == ratedItems[rIdx]) {
			subset[rIdx] = distances[idx];
			idxs[rIdx] = i+1;
			rIdx++;
		}
		idx++;
	}
}

// For one user, recommends the best K items the user hasn't rated yet
__global__ void processUser(int* recommendations, double* distances, double* sparse, int* row, int* col, int* accumRow, int numItems, int k) {

	// Parallelism: each user is assigned one block
	int userIdx = blockIdx.x;

	// For our user, search through COO to find unrated items
	int startIdx = accumRow[userIdx], numRated = accumRow[userIdx+1] - accumRow[userIdx], numUnrated = numItems - numRated;
	int *ratedIds = (int *) malloc(sizeof(int)*numRated);
	int *unratedIds = (int *) malloc(sizeof(int)*numUnrated);
	double *calculatedRatings = (double *) malloc(sizeof(double)*numUnrated);

	// Get the item IDs for our rated items
	int idx = 0, i;
	for (i = 0; i < numRated; i++) 
		ratedIds[idx++] = col[startIdx+i];

	// Get the item IDs for our unrated items
	int idx_r = 0, idx_u = 0;
	for (i = 0; i < numItems; i++)
		if (i != ratedIds[idx_r])
			unratedIds[idx_u++] = i;
		else
			idx_r++;

	// For each unrated item, find the K closest items that have been rated, take their average, and assign the unrated item a new rating
	int itemId;
	double avg;
	double *distSubset = (double *) malloc(sizeof(double)*numRated);
	int *distIdxs = (int *) malloc(sizeof(int)*numRated);
	int limit; // If we have number of rated items < k, we have to take all the rated items by default.
	if (numRated < k)
		limit = numRated;
	else
		limit = k;

	for (i = 0; i < numUnrated; i++) {
		avg = 0;
		itemId = unratedIds[i];
		distanceSubset(distances, numItems, ratedIds, numRated, itemId, distSubset, distIdxs);

		// Sort the closest-K item IDs in ascending order
		partialSort(distSubset, distIdxs, 0, numRated-1, limit, true);

		// Sort the item IDs in ascending order
		quicksort(distIdxs, 0, limit-1);

		// Get an average rating for this item
		int nIdx = 0;
		for (int j = 0; nIdx < limit; j++) {
			if (distIdxs[nIdx] == col[startIdx+j]) {
				avg += sparse[startIdx+j];
				nIdx++;
			}
		}
		calculatedRatings[i] = avg / limit;
		//calculatedRatings[i] = avg;
	}

	// Sort the calculated ratings to get the top K UNRATED items
	partialSort(calculatedRatings, unratedIds, 0, numUnrated-1, limit, false);

	// Start assigning the recommendations
	if (numUnrated < k)
		limit = numUnrated;
	else
		limit = k;
	for (i = 0; i < limit; i++)
		recommendations[userIdx*k+i] = unratedIds[i];

	free(ratedIds);
	free(unratedIds);
	free(calculatedRatings);
	free(distSubset);
	free(distIdxs);
}

// Distance pre-processor to find the top K closest items, for each item.
__global__ void ksort(double* distances, int numItems, int* topK, int k) {

	// Determine which item ID we are working with. Assuming a 1-D array of blocks
	int itemId = blockIdx.x*blockDim.x + threadIdx.x;

	// Allocate space to store the distances we wish to sort
	double* lDistances = (double *) malloc(sizeof(double)*(numItems-1));
	int* idxs = (int *) malloc(sizeof(int)*(numItems-1));

	// Start assigning distances to array
	int i, idx = 0, incrementLarge = numItems - 1, incrementSmall = itemId - 1, idxidx = itemId;
	for (i = 0; i < itemId; i++) {
		lDistances[i] = distances[idx+incrementSmall--];
		idxs[i] = i;
		idx += incrementLarge--;
	}
	for (; i < numItems; i++) {
		lDistances[i] = distances[idx++];
		idxs[idxidx++] = i+1;
	}

	// lDistances now contains the distances we want to sort, indexed in order of item order.
	// Do a partial sort of these distances and indices.
	partialSort(lDistances, idxs, 0, numItems-2, k, true);

	// Write the top K results. Now, for each item (row), we have the k most similar items (cols).
	for (i = 0; i < k; i++)
		topK[itemId*k+i] = idxs[i];

	free(lDistances);
	free(idxs);
}

/*
int main() {

	// Testing DISTANCE kernel
	// Arrays to store data and distances
	int m = 4, n = 10, num_dist = m*(m-1)/2, i, j, k = 2;
	double *data = (double *)malloc(sizeof(double)*m*n); // Data vectors, flattened
	double *distances = (double *)malloc(sizeof(double)*num_dist); // Distances, nC2 of them
	srand(time(NULL));
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++) 
			data[i*n + j] = (double) (j+i);
			//data[i*n + j] = (double) (rand()%10);

	// GPU memory to allocate
	double *d_data, *d_distances;

	// Allocate memory to GPU
	cudaMalloc((void **)&d_data,sizeof(double)*m*n);
	cudaMalloc((void **)&d_distances,sizeof(double)*num_dist);
	cudaMemcpy(d_data,data,sizeof(double)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances,distances,sizeof(double)*num_dist,cudaMemcpyHostToDevice);

	// Call kernel.
	// Only need m-1 blocks because distances go i->i+1, i->i+2, ... m-1->m
	// Only need m-1 threads for the same reason.
	calcDistance<<<m-1,m-1>>>(d_data, m, n, d_distances); // Distances only from i->j where i<j

	// Retrieve results from GPU
	cudaMemcpy(distances,d_distances,sizeof(double)*num_dist,cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	// Report
	printf("Distance kernel run - there are %i items.\n", num_dist);

	// Testing PROCESS-USER kernel
	int ldim = 14, numUsers = 3, numRated = 7;
	int *recommendations = (int *) malloc(sizeof(int)*numUsers*k);	// To store the top K recommended items for each user
	for (int i = 0; i < numUsers; i++)
		for (int j = 0; j < k; j++)
			recommendations[i*k+j] = -1;

	double sparse[] = {5, 4, 3, 3, 1, 10, 1};
	int row[] = {0, 0, 0, 1, 1, 2, 2};
	int col[] = {0, 2, 3, 1, 2, 0, 3};
	int accumRow[] = {0, 3, 5, 7};

	// Allocate to GPU
	double *d_sparse;
	int *d_recommendations, *d_row, *d_col, *d_accumRow;
	cudaMalloc((void **)&d_recommendations,sizeof(int)*numUsers*k);
	cudaMalloc((void **)&d_sparse,sizeof(double)*numRated);
	cudaMalloc((void **)&d_row,sizeof(int)*numRated);
	cudaMalloc((void **)&d_col,sizeof(int)*numRated);
	cudaMalloc((void **)&d_accumRow,sizeof(int)*(numUsers+1));
	cudaMemcpy(d_recommendations,recommendations,sizeof(int)*numUsers*k,cudaMemcpyHostToDevice);
	cudaMemcpy(d_sparse,sparse,sizeof(double)*numRated,cudaMemcpyHostToDevice);
	cudaMemcpy(d_row,row,sizeof(int)*numRated,cudaMemcpyHostToDevice);
	cudaMemcpy(d_col,col,sizeof(int)*numRated,cudaMemcpyHostToDevice);
	cudaMemcpy(d_accumRow,accumRow,sizeof(int)*(numUsers+1),cudaMemcpyHostToDevice);

	// Call kernel
	processUser<<<numUsers,1>>>(d_recommendations, d_distances, d_sparse, d_row, d_col, d_accumRow, m, k);

	// Copy results back
	cudaMemcpy(recommendations,d_recommendations,sizeof(int)*numUsers*k,cudaMemcpyDeviceToHost);

	// Report
	printf("Process-user kernel has run.\n");
	for (int i = 0; i < numUsers; i++) {
		for (int j = 0; j < k; j++)
			printf("%i\t", recommendations[i*k+j]);
		printf("\n");
	}

	int dummy;
	std::cin >> dummy;
}*/