#include<iostream>
#include<string>
#include<cstdio>
using namespace std;

#define MAX_THREADS 1024
//#define NUM_LATENTS 14 // Number of latent factors
#define RECOMMEND_K 10 // Number of items to recommend
//#define MAX_ITERATIONS 100
#define EPSILON 1e-2
//#define LAMBDA 0.01

#define UV_OFFSET(u_idx, v_idx, u_offset, v_offset, num_usrs, num_items) \
{ \
	u_offset = 2*u_idx; \
	v_offset = 2*v_idx + 1; \
	if (u_id >= num_items) { \
		u_offset = u_offset - (u_id - num_items - 1); \
	} else if (v_id >= num_usrs) { \
		v_offset = v_offset - (v_id - num_usrs - 1); \
	} \
}

bool DEV_ALLOC(void **ptr,int sz,string err_str); 
void DEV_FREE(void *ptr, string err_str); 
bool TO_DEVICE(void *d_ptr, void *ptr, int sz, string err_str); 
bool FROM_DEVICE(void *d_ptr, void *ptr, int sz, string err_str); 
void latent_modeling(double *csr_a, int *csr_ia, int *csr_ja, int *csr_ja_rows,  
	double *csc_a, int *csc_ia, int *csc_ja,
	double *u, double *v,
	double *l_u, double *l_v,
	double *l_ub, double *l_vb, 
	double *lse, int num_usrs, int num_items, int num_ratings,
	double *h_u, double *h_v,
	double *h_l_u, double *h_l_v,
	double *h_l_ub, double *h_l_vb,
	int num_iterations, double lambda); 
void recommend(double *csr_a, 
	int *csr_ia, int *csr_ja, 
	double *u, double *v, 
	double *dist_matrix, int *dist_topK, int *recommendations,
	int num_usrs, int num_items);
