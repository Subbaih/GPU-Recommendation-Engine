#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<utility>
#include<map>
#include<fstream>
#include<sstream>
#include<set>
#include<cstdio>
#include<cstring>
#include<cassert>
#include<ctime>
#include "device.h"
using namespace std;

class umatrix
{
	public:
		umatrix():rows(0),cols(0),num_elems(0),L(0),data_rwise(),data_cwise() 
		{}

		umatrix(int r, int c):rows(r),cols(c),data_rwise(r),data_cwise(r)
		{
			num_elems = 0;
			L = 0;
			cout<<"Matrix created with " << rows << " rows and "<<cols<<endl;	
		}

		void load_umatrix(int r, int c, string fname, int num_latents, char delimiter);
		bool init();
		bool alloc();
		void um_free();
		bool alloc_device();
		void free_device();
		void reload_uv();

		inline  void insert_rating(int r, int c, double rating)
		{
			data_rwise[r].push_back(make_pair(c,rating));	
			data_cwise[c].push_back(make_pair(r,rating));	
		}

		inline double operator()(int r, int c) const
		{
			for (int i=0; i<data_rwise[r].size(); i++) {
				if (data_rwise[r][i].first == c) {
					return data_rwise[r][i].second;
				}	
			}
			return 0; 
		}
	
		inline void set_u(int index,double r)
		{
			//srand(time(0));
			double min = 1, max = 5;
			for (int i=0; i<L; i++) {
				double f = (double)rand() / RAND_MAX;
				u[index*L+i] = min + f * (max-min);
			}
			fill(u+index*L,u+(index+1)*L,r);
		}
	
		inline void get_u(void *p, int index)
		{
			memcpy(p,u+index*L,L*sizeof(double));	
		}

		inline void set_v(int index,double r)
		{
			//srand(time(0));
			double min = 1, max = 5;
			for (int i=0; i<L; i++) {
				double f = (double)rand() / RAND_MAX;
				v[index*L+i] = min + f * (max-min);
			}
			fill(v+index*L,v+(index+1)*L,r);
		}
	
		inline void get_v(void *p, int index)
		{
			memcpy(p,v+index*L,L*sizeof(double));	
		}

		inline void run_modeling(int max_iterations, double lambda)
		{
			latent_modeling(d_csr_a, d_csr_ia, d_csr_ja, d_csr_ja_rows,  
				d_csc_a, d_csc_ia, d_csc_ja,
				d_u, d_v,
				d_l_u, d_l_v,
				d_l_ub, d_l_vb, 
				d_lse, rows, cols, num_elems,
				u, v,
				l_u, l_v,
				l_ub, l_vb,
				max_iterations,lambda);
		}

		inline void run_recommendation()
		{
			recommend(d_csr_a, d_csr_ia, d_csr_ja, d_u, d_v, 
				d_dist_matrix, d_dist_topK, d_recommendations, rows, cols);
			if (!FROM_DEVICE(d_recommendations,recommendations, 
				sizeof(int)*(rows*RECOMMEND_K),"d_recommendations")) {
				printf("Unable to fetch the recommendations from device\n");
				return;
			}
			printf("Following are the recommendations:\n");
			for (int i = 0; i < rows; i++) {
				printf("User-%d ",i);
				for (int j = 0; j < RECOMMEND_K; j++) {
					printf("%i\t", recommendations[i*RECOMMEND_K+j]);
				}
				printf("\n");
			}
		}

	private:
		int rows;
		int cols;
		int num_elems;
		int L;
		/* Utility matrix sparse representation */
		double *csr_a, *d_csr_a, *csc_a, *d_csc_a;
		int *csr_ia, *d_csr_ia, *csr_ja, *d_csr_ja, *csr_ja_rows, *d_csr_ja_rows; 
		int *csc_ia, *d_csc_ia, *csc_ja, *d_csc_ja;
		/* Latent Vectors */
		double *u, *d_u, *v, *d_v;

		/* Temporary variables */ 
		/* Matrices for temporary holdings during factorization */	
		double *l_u, *d_l_u;
		double *l_v, *d_l_v;
		/* Vectors for temporary holdings during factorization */	
		double *l_ub, *d_l_ub;
		double *l_vb, *d_l_vb;
		/* least squared error for each rating */
		double *lse, *d_lse;
		double *dist_matrix, *d_dist_matrix;
		int *dist_topK, *d_dist_topK;
		int *recommendations, *d_recommendations;
	public:
		vector<vector<pair<int,double> > > data_rwise;
		vector<vector<pair<int,double> > > data_cwise;
};

