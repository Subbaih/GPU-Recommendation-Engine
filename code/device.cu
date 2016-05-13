#include<cuda.h>
#include <cmath>
#include <cfloat>
#include "device.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <thrust\device_vector.h>
//#include <thrust\find.h>
//#include <thrust\execution_policy.h>

extern int num_latents;

#define INT_CEIL(a,b) \
	((a)/(b) + (((a)%(b)) ? 1 : 0))

// Blocks in 1 Dimension; Threads in 1 Dimension
#define GET_GLOBAL_IDX_1D_1D(blockId,threadId) \
	blockId = blockIdx.x; \
	threadId = blockId * blockDim.x  + threadIdx.x; 

// Blocks in 1 Dimension; Threads in 2 Dimensions
#define GET_GLOBAL_IDX_1D_2D(blockId,threadId) \
	blockId = blockIdx.x; \
	threadId = blockId * (blockDim.x * blockDim.y) + \
	(threadIdx.y * blockDim.x) + threadIdx.x;

#define CREATE_EVENT(event,err_str) \
{\
	cudaError_t err = cudaEventCreate(&event);\
	if (err != cudaSuccess) {\
	cout<<"cudaEventCreate failed for "<< err_str<<" reason:"<<err<<endl;\
	}\
}

#define RECORD_EVENT(event,err_str,sync) \
{\
	cudaError_t err = cudaEventRecord(event,0);\
	if (err != cudaSuccess) {\
	cout<<"cudaEventRecord failed for "<< err_str<<" reason:"<<err<<endl;\
	}\
	if (sync) {\
	err = cudaEventSynchronize(event);\
	if (err != cudaSuccess) {\
	cout<<"cudaEventSynchronize failed for "<< err_str<<" reason:"<<err<<endl;\
	}\
	}\
} 

#define ELAPSED_TIME(time,e_start,e_stop,err_str) \
{\
	cudaError_t err = cudaEventElapsedTime(&time,start,stop);\
	if (err != cudaSuccess) { \
	cout<<"cudaEventElapsedTime failed for "<< err_str<<" reason:"<<err<<endl;\
	} else { \
	cout<<"Time taken for "<<err_str<<" is "<<time<<endl; \
	}\
}

#define PU_THREADS 50
#define SIZE_K 10

bool DEV_ALLOC(void **ptr, int sz, string err_str) 
{
	cudaError_t err;
	err = cudaMalloc(ptr,sz); 
	if (err != cudaSuccess) { 
		cout<<err_str<<" device_alloc "<<sz<<" failed:"<<err<<endl;
		return false; 
	} 
	return true;
}

void DEV_FREE(void *ptr, string err_str) 
{
	cudaError_t err = cudaFree(ptr); 
	if (err != cudaSuccess) { 
		cout<<err_str<<" device_free  failed:"<<err<<endl;
	} 
}

bool TO_DEVICE(void *d_ptr, void *ptr, int sz, string err_str) 
{
	cudaError_t err = cudaMemcpy(d_ptr,ptr,sz,cudaMemcpyHostToDevice); 
	if (err != cudaSuccess) { 
		cout<<err_str<<" transfer-to-device "<<sz<<" failed:"<<err<<endl;
		return false; 
	} 
	return true;
}

bool FROM_DEVICE(void *d_ptr, void *ptr, int sz, string err_str) 
{
	cudaError_t err = cudaMemcpy(ptr,d_ptr,sz,cudaMemcpyDeviceToHost); 
	if (err != cudaSuccess) { 
		cout<<err_str<<" transfer-from-device "<<sz<<" failed:"<<err<<endl;
		return false; 
	} 
	return true;
}

__global__ void entities_multiply_kernel(double *cs_a, int *cs_ia, int *cs_ja, 
	double *entities_vec_base, double *l_vec_base, double *lb_vec_base, 
	int num_latents, double lambda, int max_threads) 
{
	int b_idx, idx;
	GET_GLOBAL_IDX_1D_1D(b_idx,idx); 
	if(idx >= max_threads) {
		return;
	}

	int num_rated = cs_ia[idx+1] - cs_ia[idx];
	int cs_ja_offset = cs_ia[idx];
	double* result = l_vec_base + (idx * num_latents * num_latents);
	memset(result,0,num_latents*num_latents*sizeof(double));
	double* lb_result = lb_vec_base + (idx * num_latents);
	memset(lb_result,0,num_latents*sizeof(double));

	//printf("idx=%d num_rated=%d\n",idx,num_rated);
	for (int n=0; n<num_rated; n++) {
		int entity_id = cs_ja[cs_ja_offset+n];
		double* V = entities_vec_base + (entity_id * num_latents);
		for (int i=0; i<num_latents; i++) {
			result[i*num_latents+i]+=V[i]*V[i];
			for (int j=i+1; j<num_latents; j++) {
				result[i*num_latents+j]+=V[i]*V[j];
				result[j*num_latents+i]+=V[i]*V[j];
			}
		}

		double rating = cs_a[cs_ja_offset+n];
		for (int i=0; i<num_latents; i++) {
			lb_result[i]  += (V[i]*rating);
			//printf("idx=%d i=%d rating=%g\n",idx,i,V[i]);
		}
	}

	for (int i=0; i<num_latents; i++) {
		result[i*(num_latents+1)]+= (lambda*num_rated);
	}
	//cudaDeviceSynchronize();
}

// Calculates the mean squared error for actual and predicted ratings
__global__ void predict_ratings_kernel(double *csr_a, 
	int *csr_ja_rows, int *csr_ja, 
	double *u_base, double *v_base, double *sle, 
	int num_latents, int max_threads)
{
	int t_idx, b_idx;
	GET_GLOBAL_IDX_1D_1D(b_idx,t_idx); 
	if (t_idx >= max_threads)
		return;

	// Find the row and column
	int u_id = csr_ja_rows[t_idx], v_id = csr_ja[t_idx];
	double *u = u_base + (u_id*num_latents);
	double *v = v_base + (v_id*num_latents);

	// Calculate the dot product between u and v
	double s = 0.0;
	for (int i=0; i<num_latents; i++) {
		s += (u[i]*v[i]);
		//if (b_idx <= 1)
		//printf("u_id=%g v_id=%g s=%g\n",u[i],v[i],s);
	}
	double err = pow((s-csr_a[t_idx]),2);
	sle[t_idx] = err; 
}

__global__ void mse_kernel(double *ip_base, double *op_base, 
	int workload, int l_workload, int max_threads)
{
	int t_idx, b_idx;
	GET_GLOBAL_IDX_1D_1D(b_idx,t_idx); 
	if (t_idx >= max_threads) {
		return;
	} 

	double *ip = ip_base + (t_idx * workload);
	if (t_idx == max_threads-1) {
		workload = l_workload;	
		//printf("t_idx=%d lworkload=%d\n",t_idx,l_workload);
	}	
	double s = 0.0;
	for (int i=0; i<workload; i++) {
		s  = s + ip[i];
	}
	//printf("t_idx=%d s=%g\n",t_idx,s);
	*(op_base + t_idx) = s/workload;
} 

// Solves the Linear Equations Ax = B 
// using LDL decomposition of symmetric matrix A
#define A(a,i,j,N) (*((a)+i*N+j))
__global__ void LDL_solve_kernel(double *s_a, double *s_x, 
	double *s_b, int nl, int max_threads)
{
	int t_idx, b_idx;
	GET_GLOBAL_IDX_1D_1D(b_idx,t_idx); 
	if (t_idx >= max_threads)
		return;

	double *a = s_a + (nl * nl * t_idx);
	double *x = s_x + (nl * t_idx);
	double *b = s_b + (nl * t_idx);
	double temp;

	for (int i=0; i<nl; i++) {
		for (int j=0; j<i; j++) {
			temp = 0.0;
			for (int k=0; k<j; k++) {
				temp+=(A(a,i,k,nl)*A(a,k,k,nl)*A(a,j,k,nl));	
			}
			A(a,i,j,nl) = (A(a,i,j,nl) - temp)/A(a,j,j,nl);
		}
		temp = 0.0;
		for (int k=0; k<i; k++) {
			temp += (A(a,i,k,nl)*A(a,i,k,nl)*A(a,k,k,nl));  
		}
		A(a,i,i,nl) = A(a,i,i,nl) - temp;
		if (A(a,i,i,nl) == 0) {
			A(a,i,i,nl) = DBL_EPSILON;
		}
	}

	// Forward Substitution
	for (int i=0; i<nl; i++) {
		temp = 0.0;
		for (int j=0; j<i; j++) {
			temp+=(A(a,i,j,nl)*x[j]);
		} 
		x[i] = (b[i] - temp);
	}	

	// Backward Substitution
	for (int i=nl-1; i>=0; i--) {
		temp = 0.0;
		for (int j=i+1; j<nl; j++) {
			temp+=(A(a,j,i,nl)*x[j]);
		} 
		x[i] = x[i]/A(a,i,i,nl) - temp;
	}
}

__global__ void calcDistance(const double* __restrict__ data, int m, int n, double* __restrict__ distances)
{

	int item_id, compare_id, global_idx;
	int  global_idx_start = 0, global_idx_limit, i, j;
	double value;
	
	item_id = blockIdx.x;	// Each block calculates the distances from ONE item to the rest of the items.
	compare_id = item_id + 1 + threadIdx.x; // Each thread calculates the distance from ONE item to ONE item.
	// AKA: what's the item ID we are calculating the distance to?

	// Find starting global idx in distances array
	global_idx_limit = m - 1;
	for (i = 0; i < item_id; i++) {
		global_idx_start += (m - 1 - i);
		global_idx_limit--;
	}

	global_idx = global_idx_start + (compare_id - item_id) - 1;
	global_idx_limit = global_idx_start + global_idx_limit;	// We cannot go beyond this index

	while (compare_id < m && global_idx < global_idx_limit) {
		value = 0.0;

		// Calculate distance
		for (j = 0; j < n; j++) // Iterates through dimensions of one data row
			value += pow(data[compare_id*n + j] - data[item_id*n + j], 2);

		distances[global_idx] = value;
		compare_id += PU_THREADS;
		global_idx = global_idx_start + (compare_id - item_id) - 1;		// Where in the distances array are we writing next?
	}

}

// Quicksort partition function
__device__ int partition(double* __restrict__ distances, int* __restrict__ idxs, int left, int right) {
	double pivot = distances[right], dummyDistance;
	int newPivotIdx = left, dummyIdx;
	for (int i = left; i < right; i++) {
		if (distances[i] < pivot) {
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

// Quicksort partition function, only for item IDs and their ratings
__device__ int partition(int* __restrict__ idxs, double* __restrict__ ratings, int left, int right) {
	int pivot = idxs[right], newPivotIdx = left, dummyIdx;
	double dummyRating;
	for (int i = left; i < right; i++) {
		if (idxs[i] < pivot) {
			dummyRating = ratings[newPivotIdx];
			ratings[newPivotIdx] = ratings[i];
			ratings[i] = dummyRating;
			dummyIdx = idxs[newPivotIdx];
			idxs[newPivotIdx] = idxs[i];
			idxs[i] = dummyIdx;
			newPivotIdx++;
		}
	}
	dummyRating = ratings[newPivotIdx];
	ratings[newPivotIdx] = ratings[right];
	ratings[right] = dummyRating;
	dummyIdx = idxs[newPivotIdx];
	idxs[newPivotIdx] = idxs[right];
	idxs[right] = dummyIdx;

	return newPivotIdx;
}

// Quicksort function
__device__ void quicksort(double* __restrict__ distances, int* __restrict__ ids, int left, int right) {

	while (left < right) {
		int newPivotIdx = partition(distances, ids, left, right);
		if (newPivotIdx - left > right - newPivotIdx) {
			quicksort(distances, ids, newPivotIdx+1, right);
			right = newPivotIdx - 1;
		}
		else {
			quicksort(distances, ids, left, newPivotIdx-1);
			left = newPivotIdx + 1;
		}
	}
}

__device__ void quicksortIter(double* __restrict__ distances, int* __restrict__ idxs, int left, int right) {
	// Stack for recursive calls
	//int *stack = (int *) malloc(sizeof(int)*(right-left+1));
	int stack[100];

	// initialize top of stack
	int top = 0, newPivotIdx;

	// push args onto stack
	stack[top++] = left;
	stack[top++] = right;

	// Keep popping from stack while is not empty
	while ( top >= 0 ) {
		// Pop h and l
		right = stack[--top];
		left = stack[--top];

		// Call partition function
		if (left >= 0 && left < right) {
			newPivotIdx = partition(distances, idxs, left, right);

			// If there are elements on left side of pivot,
			// then push left side to stack

			if ( newPivotIdx-1 > left ) {
				stack[ top++ ] = left;
				stack[ top++ ] = newPivotIdx - 1;
			}

			// If there are elements on right side of pivot,
			// then push right side to stack
			if ( newPivotIdx+1 < right ) {
				stack[top++] = newPivotIdx + 1;
				stack[top++] = right;
			}
		}
	}
}

// Iterative quicksort, for item IDs and their ratings
__device__ void quicksortIter(int* __restrict__ idxs, double* __restrict__ ratings, int left, int right) {
	// Stack for recursive calls
	//int *stack = (int *) malloc(sizeof(int)*(right-left+1));
	int stack[100];

	// initialize top of stack
	int top = 0, newPivotIdx;

	// push args onto stack
	stack[top++] = left;
	stack[top++] = right;

	// Keep popping from stack while is not empty
	while ( top >= 0 ) {
		// Pop h and l
		right = stack[--top];
		left = stack[--top];

		// Call partition function
		if (left >= 0 && left < right) {
			newPivotIdx = partition(idxs, ratings, left, right);

			// If there are elements on left side of pivot,
			// then push left side to stack

			if ( newPivotIdx-1 > left ) {
				stack[ top++ ] = left;
				stack[ top++ ] = newPivotIdx - 1;
			}

			// If there are elements on right side of pivot,
			// then push right side to stack
			if ( newPivotIdx+1 < right ) {
				stack[top++] = newPivotIdx + 1;
				stack[top++] = right;
			}
		}
	}
}

// Partial-k sort
__device__ void partialSort(double* __restrict__ distances, 
	int* __restrict__ idxs, int left, int right, int k) {
		int newPivotIdx;
		while (left < right) {
			newPivotIdx = partition(distances, idxs, left, right);
			if (k < newPivotIdx) {
				right = newPivotIdx - 1;
			} else if (newPivotIdx - left > right - newPivotIdx) {
				quicksortIter(distances, idxs, newPivotIdx+1, right);
				right = newPivotIdx - 1;
			} else {
				quicksortIter(distances, idxs, left, newPivotIdx-1);
				left = newPivotIdx + 1;
			}
		}
}

// Gets a subset of distances between a given item and a set of rated items
__device__ void distanceSubset(const double* __restrict__ distances, 
	int num_items, int* ratedItems, 
	int num_ratedItems, int itemId, 
	double* __restrict__ subset, int* __restrict__ idxs) {
		int i, idx = 0, rIdx = 0, incrementLarge = num_items - 1, incrementSmall = itemId - 1;

		for (i = 0; i < itemId && rIdx < num_items; i++) {
			if (i == ratedItems[rIdx]) {
				subset[rIdx] = distances[idx+incrementSmall];
				idxs[rIdx] = i;
				rIdx++;
			}
			incrementSmall--;
			idx += incrementLarge--;
		}
		for (; i < num_items && rIdx < num_items; i++) {
			if (i+1 == ratedItems[rIdx]) {
				subset[rIdx] = distances[idx];
				rIdx++;
			}
			idx++;
		}
}

// Distance pre-processor to sort item distances, for each item
__global__ void presort(const double* __restrict__ distances, int numItems, int* __restrict__ sorted) {

	int ni = numItems*(numItems-1)/2;
	// Determine which item ID we are working with. Assuming a 1-D array of blocks

	int itemId = blockIdx.x*blockDim.x + threadIdx.x;

	// Allocate space for temporary storage
	double* __restrict__ lDistances = (double *) malloc(sizeof(double)*(numItems-1));
	int* __restrict__ idxs = (int *) malloc(sizeof(int)*(numItems-1));

	if (itemId < numItems) {

		// Start assigning distances to array
		int i, idx = 0, incrementLarge = numItems - 1, incrementSmall = itemId - 1, idxidx = itemId;
		for (i = 0; i < itemId && idx < ni; i++) {
			lDistances[i] = distances[idx+incrementSmall--];
			idxs[i] = i;
			idx += incrementLarge--;
		}
		for (; i < numItems-1 && idxidx < ni; i++) {
			lDistances[i] = distances[idx++];
			idxs[idxidx++] = i+1;
		}

		// lDistances now contains the distances we want to sort, indexed in order of item order.
		// Do a sort of these distances and indices.
		quicksortIter(lDistances, idxs, 0, numItems-2);

		// Write the results back to the results array
		for (i = 0; i < numItems-1; i++)
			sorted[itemId*(numItems-1)+i] = idxs[i];

		printf("%i\t", itemId);
	}
	free(lDistances);
	free(idxs);
}

// Get the closest K items to a particular item ID (that has not been rated); the K items must intersect the set of rated items
__device__ void closestK(const int* __restrict__ presortedIds, int num_items, 
	int* __restrict__ ratedIds, double* __restrict__ ratedRatings, int num_rated, int itemId, int* __restrict__ dIdxSubset, double* __restrict__ ratingSubset, int k) {

		// Find index
		int idxBegin = itemId*(num_items-1);

		//// Until k, try to match each presorted ID to our valid IDs
		//int curId, rIdx = 0;
		//for (int i = 0; rIdx < k && i < num_items-1; i++) {
		//	curId = presortedIds[idxBegin+i];
		//	if (thrust::find(thrust::device,ratedIds,ratedIds+k,curId) - ratedIds < k) {
		//		dIdxSubset[rIdx] = curId;
		//		ratingSubset[rIdx] = ratedRatings[rIdx];
		//		rIdx++;
		//	}
		//}
		
		int ids[SIZE_K];
		//thrust::copy(thrust::device,ratedIds,ratedIds+10,ids);
		for (int i = 0; i < SIZE_K; i++)
			ids[i] = ratedIds[i];

		// Go down the presorted list and check for membership in the rated IDs list. This is inefficient; replace with proper implementation later
		int i = 0, j = 0, rIdx = 0, jumpstart = 0;
		while (i < num_items-1 && rIdx < k && rIdx < num_rated) {
			for (j = jumpstart; j < k; j++)  // Check membership
				if (ids[j] == presortedIds[idxBegin+i]) {
					dIdxSubset[rIdx] = presortedIds[idxBegin+i];
					ratingSubset[rIdx] = ratedRatings[rIdx];
					
					ids[j] = ids[jumpstart];
					jumpstart++;
					rIdx++;
					break;
				}
			
			i++;
		}
}

// Presorted matrix of ITEMS x ITEMS where each row has an ordered array of closest items.
__global__ void processUser(int* __restrict__ recommendations, const int* __restrict__ presortedIds, 
	const double* __restrict__ csr_a, const int* __restrict__ csr_ia, const int* __restrict__ csr_ja, int num_items, int k) 
{

	// Parallelism: each user is assigned one block
	int userIdx = blockIdx.x;

	// For our user, search through COO to find unrated items
	int startIdx = csr_ia[userIdx];
	int num_rated = csr_ia[userIdx+1] - csr_ia[userIdx];
	int num_unrated = num_items - num_rated;
	int* __restrict__ ratedIds = (int *) malloc(sizeof(int)*num_rated);
	int* __restrict__ unratedIds = (int *) malloc(sizeof(int)*num_unrated);
	double* __restrict__ ratedRatings = (double *) malloc(sizeof(double)*num_rated); // To store user ratings

	//double *calculatedRatings = (double *) malloc(sizeof(double)*num_unrated);
	__shared__ double calculatedRatings[2000];

	// Get the item IDs for our rated items (and their ratings)
	int idx = 0, i;
	for (i = 0; i < num_rated; i++) {
		ratedIds[idx] = csr_ja[startIdx+i];
		ratedRatings[idx] = csr_a[startIdx+i];
		idx++;
	}

	// Get the item IDs for our unrated items
	int idx_u = 0;
	for (i = 0; i < num_items; i++) {
		bool found = false;
		for (int j=0; j < num_rated; j++) {
			if (ratedIds[j] == i) {
				found = true;
				break;
			}
		}
		if (!found) {
			unratedIds[idx_u++] = i;
		} 
	}

	quicksortIter(ratedIds,ratedRatings,0,num_rated-1); // Sort the rated IDs in ascending order; also sort the ratings values array

	// For each unrated item, find the K closest items that have been rated, 
	// take their average, and assign the unrated item a new rating.
	// Let each thread take one item ID
	int avg, itemId = -1, cIdx = -1;
	if (threadIdx.x < num_unrated)
		cIdx = threadIdx.x;

	//double* __restrict__ dSubset = (double *) malloc(sizeof(double)*(num_items-1));
	int* __restrict__ dIdxSubset = (int *) malloc(sizeof(int)*k);
	double* __restrict__ ratingSubset = (double *) malloc(sizeof(double)*k);

	while (cIdx >= 0 && cIdx < num_unrated) {
		itemId = unratedIds[cIdx];
		avg = 0;

		// Take only the cloest K rated items to this unrated item
		closestK(presortedIds, num_items, ratedIds, ratedRatings, num_rated, itemId, dIdxSubset, ratingSubset, k);

		// Get an average rating for this item
		for (int j = 0; j < k; j++) {
			avg += ratingSubset[j];
		}
		calculatedRatings[cIdx] = avg / k;

		cIdx += PU_THREADS;
	}

	free(ratedIds);
	free(ratedRatings);
	free(ratingSubset);
	//free(dSubset);
	free(dIdxSubset);
	__syncthreads();


	if (threadIdx.x == 0) {
		// Sort the calculated ratings to get the top K items
		partialSort(calculatedRatings, unratedIds, 0, num_unrated-2, k);
		printf("%i\t", userIdx);

		// Return the top K items
		for (i = 0; i < k; i++) {
			recommendations[userIdx*k+i] = unratedIds[i];
		}
	}

	free(unratedIds);
}

void print_vector(double *v, int len, int dim, string prefix)
{
	int r = 0;
	cout<<prefix<<endl;
	for (int l=0; l<len; l++) {
		for (int i=0; i<dim; i++) {
			printf("%g ",v[r++]);
		}
		printf("\n");	
	}
	printf("\n");	
}

void print_matrix(double *m, int len, int dim1, int dim2, string prefix)
{
	int r = 0;	
	cout<<prefix<<endl;
	for (int l=0; l<len; l++) {
		for (int i=0; i<dim1; i++) {
			for (int i=0; i<dim2; i++) {
				printf("%g ",m[r++]);
			}
		}
		printf("\n");	
	}
	printf("\n");	
}

void latent_modeling(double *csr_a, int *csr_ia, int *csr_ja, int *csr_ja_rows,  
	double *csc_a, int *csc_ia, int *csc_ja,
	double *u, double *v,
	double *l_u, double *l_v,
	double *l_ub, double *l_vb, 
	double *lse, int num_usrs, int num_items, int num_ratings,
	double *h_u, double *h_v,
	double *h_l_u, double *h_l_v,
	double *h_l_ub, double *h_l_vb,
	int max_iterations, double lambda) 
{
	printf("num-ratings=%d\n",num_ratings);
	cudaEvent_t start, stop;
	CREATE_EVENT(start,"General-Start event");
	CREATE_EVENT(stop,"General-Stop event");
	float time;

	double mserr;
	int max_threads = MAX_THREADS;
	int num_blocks = 0;

	num_blocks = INT_CEIL(num_usrs,max_threads);
	dim3 step1_thread(max_threads,1);
	dim3 step1_block(num_blocks,1,1);
	int step1_max_threads = num_usrs;

	num_blocks = INT_CEIL(num_usrs,max_threads);
	dim3 step2_thread(max_threads,1,1);
	dim3 step2_block(num_blocks,1,1);
	int step2_max_threads = num_usrs;

	num_blocks = INT_CEIL(num_items,max_threads);
	dim3 step3_thread(max_threads,1,1);
	dim3 step3_block(num_blocks,1,1);
	int step3_max_threads = num_items;

	num_blocks = INT_CEIL(num_items,max_threads);
	dim3 step4_thread(max_threads,1,1);
	dim3 step4_block(num_blocks,1,1);
	int step4_max_threads = num_items;

	dim3 step5_1_thread(max_threads,1,1);
	num_blocks = INT_CEIL(num_ratings,max_threads);
	dim3 step5_1_block(num_blocks,1,1);
	int step5_1_max_threads = num_ratings;
	printf("step5_1 blocks=%d threads-per-block=%d "
		"max_threads=%d workload=1\n", num_blocks,
		max_threads, step5_1_max_threads);

	double *l_max_entity_b; 
	int step5_2_max_threads;
	int step5_2_workload, step5_2_lworkload;
	if (num_usrs >= num_items) {
		l_max_entity_b = l_ub;
		num_blocks = INT_CEIL(num_usrs,max_threads);
		step5_2_max_threads = num_usrs;
		step5_2_workload = num_ratings/num_usrs;
		step5_2_lworkload = num_ratings%num_usrs;
	} else {
		l_max_entity_b = l_vb;
		num_blocks = INT_CEIL(num_items,max_threads);
		step5_2_max_threads = num_items;
		step5_2_workload = num_ratings/num_items;
		step5_2_lworkload = num_ratings%num_items;
	}
	dim3 step5_2_thread(max_threads,1,1);
	dim3 step5_2_block(num_blocks,1,1);
	printf("step5_2 blocks=%d threads-per-block=%d "
		"max_threads=%d workload=%d l_workload=%d\n", num_blocks,
		max_threads, step5_2_max_threads, step5_2_workload, step5_2_lworkload);

	dim3 step5_3_thread(1,1,1);
	dim3 step5_3_block(1,1,1);
	int step5_3_max_threads = 1;
	int step5_3_workload = step5_2_max_threads;
	printf("step5_3 workload=%d\n",step5_3_workload);

	for (int i=0; i<max_iterations; i++) {
		// 1) Calculate (V_i)(V_i^T) + LAMBDA*I - num_latents*num_latents matrix
		// printf("step-1\n");
		RECORD_EVENT(start,"Step-1",0);
		entities_multiply_kernel<<<step1_block,step1_thread>>>(
			csr_a, csr_ia, csr_ja, v, l_u, l_ub, num_latents, lambda,
			step1_max_threads);
		RECORD_EVENT(stop,"Step-1",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-1");
		/*
		FROM_DEVICE(l_u,h_l_u,sizeof(double)*num_usrs*num_latents*num_latents,"h_l_u");
		FROM_DEVICE(l_ub,h_l_ub,sizeof(double)*num_usrs*num_latents,"h_l_ub");
		print_matrix(h_l_u,num_usrs,num_latents,num_latents,"l_u");
		print_vector(h_l_ub,num_usrs,num_latents,"l_ub");
		*/		

		// printf("step-2\n");
		// 2) Solve u_i = [(V_i)(V_i^T) + LAMBDA*I]^(-1) [(V_i)(R_i) - (num_latents*nu_i)*(nu_i*1) matrix]
		RECORD_EVENT(start,"Step-2",0);
		LDL_solve_kernel<<<step2_block,step2_thread>>>(l_u, u, l_ub, num_latents, step2_max_threads);
		RECORD_EVENT(stop,"Step-2",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-2");
		/*
		FROM_DEVICE(l_u,h_l_u,sizeof(double)*num_usrs*num_latents*num_latents,"h_l_u");
		FROM_DEVICE(l_ub,h_l_ub,sizeof(double)*num_usrs*num_latents,"h_l_ub");
		FROM_DEVICE(u,h_u,sizeof(double)*num_usrs*num_latents,"h_u");
		print_matrix(h_l_u,num_usrs,num_latents,num_latents,"l_u");
		print_vector(h_l_ub,num_usrs,num_latents,"l_ub");
		print_vector(h_u,num_usrs,num_latents,"u");
		*/

		// 3) Calculate (U_j)(U_j^T) + LAMBDA*I - num_latents*num_latents matrix
		// printf("step-3\n");
		RECORD_EVENT(start,"Step-3",0);
		entities_multiply_kernel<<<step3_block,step3_thread>>>(
			csc_a, csc_ia, csc_ja, u, l_v, l_vb, num_latents, lambda,
			step3_max_threads);
		RECORD_EVENT(stop,"Step-3",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-3");
		/*
		FROM_DEVICE(l_v,h_l_v,sizeof(double)*num_items*num_latents*num_latents,"h_l_v");
		FROM_DEVICE(l_vb,h_l_vb,sizeof(double)*num_items*num_latents,"h_l_vb");
		print_matrix(h_l_v,num_items,num_latents,num_latents,"l_v");
		print_vector(h_l_vb,num_items,num_latents,"l_vb");
		*/

		// printf("step-4\n");
		// 4) Solve v_j = [(U_j)(U_j^T) + LAMBDA*I]^(-1) [(U_j)(R_j) - (num_latents*nv_j)*(nv_j*1) matrix]
		RECORD_EVENT(start,"Step-4",0);
		LDL_solve_kernel<<<step4_block,step4_thread>>>(l_v, v, l_vb, num_latents, step4_max_threads);
		RECORD_EVENT(stop,"Step-4",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-4");
		/*
		FROM_DEVICE(l_v,h_l_v,sizeof(double)*num_items*num_latents*num_latents,"h_l_v");
		FROM_DEVICE(l_vb,h_l_vb,sizeof(double)*num_items*num_latents,"h_l_vb");
		FROM_DEVICE(v,h_v,sizeof(double)*num_items*num_latents,"h_v");
		print_matrix(h_l_v,num_items,num_latents,num_latents,"l_v");
		print_vector(h_l_vb,num_items,num_latents,"l_vb");
		print_vector(h_v,num_items,num_latents,"v");
		*/

		// 5) Calculate Squared error between R and U*V for original ratings - Reduction
		// a) 1D blocks in 2D threads
		RECORD_EVENT(start,"Step-5a",0);
		predict_ratings_kernel<<<step5_1_block,step5_1_thread>>>(
			csr_a, csr_ja_rows, csr_ja, 
			u, v, lse, num_latents, step5_1_max_threads);
		RECORD_EVENT(stop,"Step-5a",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-5a");

		// b) 1D blocks in 1D max_threads
		RECORD_EVENT(start,"Step-5b",0);
		mse_kernel<<<step5_2_block,step5_2_thread>>>(lse, l_max_entity_b,
			step5_2_workload, step5_2_lworkload, step5_2_max_threads);
		RECORD_EVENT(stop,"Step-5b",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-5b");

		// c) 1D blocks in 1D max_threads
		RECORD_EVENT(start,"Step-5c",0);
		mse_kernel<<<step5_3_block,step5_3_thread>>>(l_max_entity_b, lse,
			step5_3_workload, step5_3_workload, step5_3_max_threads);
		RECORD_EVENT(stop,"Step-5c",0);
		cudaDeviceSynchronize();
		ELAPSED_TIME(time,start,stop,"Step-5c");
		if (!FROM_DEVICE(lse, &mserr, sizeof(mserr), "Iteration-Error")) {
			break; 
		} else {
			printf("Iteration-%d error:%g\n",i,mserr);
			if (mserr <= EPSILON) {
				printf("Acceptable error of %g reached at Iteration-%d\n",mserr,i);
				break;
			}
		}
	}
}

void recommend(double *csr_a, 
	int *csr_ia, int *csr_ja, 
	double *u, double *v, 
	double *dist_matrix, int *dist_topK, int *recommendations,
	int num_usrs, int num_items)
{
	cudaEvent_t start, stop;
	CREATE_EVENT(start,"General-Start event");
	CREATE_EVENT(stop,"General-Stop event");
	float time;

	RECORD_EVENT(start,"calcDistance",0);
	calcDistance<<<num_items-1,PU_THREADS>>>(v, num_items, num_latents, dist_matrix);
	RECORD_EVENT(stop,"calcDistance",0);
	cudaDeviceSynchronize();
	ELAPSED_TIME(time,start,stop,"calcDistance");

	// Free some device memory for next kernel
	cudaFree(u);
	cudaFree(v);

	// Presort distances
	int* sorted;
	cudaMalloc((void **)&sorted, sizeof(int)*num_items*(num_items-1));
	presort<<<(num_items-1)/PU_THREADS+1,PU_THREADS>>>(dist_matrix, num_items, sorted);
	cudaDeviceSynchronize();

	// Free some device memory for next kernel
	cudaFree(dist_matrix);

	RECORD_EVENT(start,"processUser",0);
	processUser<<<num_usrs,PU_THREADS>>>(recommendations, sorted, 
		csr_a, csr_ia, csr_ja, num_items, RECOMMEND_K);
	RECORD_EVENT(stop,"processUser",0);
	cudaDeviceSynchronize();
	ELAPSED_TIME(time,start,stop,"processUser");

	return;	
}
