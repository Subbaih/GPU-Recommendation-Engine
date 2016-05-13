#include "umatrix.h"

#define HOST_ALLOC_N_CLEAR(ptr,type,sz,err_str) \
    {\
        ptr = new type[sz]; \
        if (ptr == NULL) { \
            cout<<err_str<<" alloc "<<sz<<" failed"<<endl;\
	    goto ErrorHandler; \
        } \
        memset(ptr,0,sz*sizeof(type)); \
    }\
 

void umatrix::load_umatrix(int r, int c, string fpath, 
	int num_latents, char delimiter) 
{
	rows = r;
	cols = c;
	num_elems = 0;
	L = num_latents;
	data_rwise.resize(rows);
	data_cwise.resize(cols);
	string line;
	string fname = fpath + "u.data";
	ifstream infile(fname.c_str());
	while (getline(infile,line)) {
		istringstream iss(line.c_str());
		int user_id, item_id;
		double rating;
		string s1_id, s2_id, s3_id;
		getline(iss,s1_id,delimiter); istringstream iss1(s1_id.c_str()); iss1>>user_id; 
		getline(iss,s2_id,delimiter); istringstream iss2(s2_id.c_str()); iss2>>item_id; 
		getline(iss,s3_id,delimiter); istringstream iss3(s3_id.c_str()); iss3>>rating; 
		//cout<<" U="<<user_id<<" I="<<item_id<<" R="<<rating<<endl;
		assert(user_id<=r);
		assert(item_id<=c);
		insert_rating(user_id-1, item_id-1, rating);
		num_elems++;
	}
	cout<<"umatrix constructor done"<<endl;
}

bool umatrix::alloc()
{
	cout<<"umatrix::alloc() called"<<endl;
	HOST_ALLOC_N_CLEAR(csr_a,double,num_elems,"csr_a");
	HOST_ALLOC_N_CLEAR(csr_ia,int,(rows+1),"csr_ia");
	HOST_ALLOC_N_CLEAR(csr_ja,int,num_elems,"csr_ja");
	HOST_ALLOC_N_CLEAR(csr_ja_rows,int,num_elems,"csr_ja_rows");
	HOST_ALLOC_N_CLEAR(csc_a,double,num_elems,"csc_a");
	HOST_ALLOC_N_CLEAR(csc_ia,int,(cols+1),"csc_ia");
	HOST_ALLOC_N_CLEAR(csc_ja,int,num_elems,"csc_ja");
	HOST_ALLOC_N_CLEAR(u,double,(L*rows),"u");
	HOST_ALLOC_N_CLEAR(v,double,(L*cols),"v");
	HOST_ALLOC_N_CLEAR(l_u,double,(L*L*rows),"l_u");
	HOST_ALLOC_N_CLEAR(l_v,double,(L*L*cols),"l_v");
	HOST_ALLOC_N_CLEAR(l_ub,double,(L*rows),"l_ub");
	HOST_ALLOC_N_CLEAR(l_vb,double,(L*cols),"l_vb");
	HOST_ALLOC_N_CLEAR(lse,double,num_elems,"lse");
	HOST_ALLOC_N_CLEAR(dist_matrix,double,(cols)*(cols-1)/2,"dist_matrix");
	HOST_ALLOC_N_CLEAR(dist_topK,int,(cols*RECOMMEND_K),"dist_topK");
	HOST_ALLOC_N_CLEAR(recommendations,int,(rows*RECOMMEND_K),"recommendations");
	cout<<"umatrix::alloc() call ended"<<endl;
	return true;
ErrorHandler:
	um_free();
	return false;
}

void umatrix::um_free()
{
	cout<<"umatrix::um_free() called"<<endl;
	//printf("csr_a=%p\n",csr_a);
	delete(csr_a);	
	//printf("csr_ia=%p\n",csr_ia);
	delete(csr_ia);	
	//printf("csr_ja=%p\n",csr_ja);
	delete(csr_ja);	
	//printf("csr_ja_rows=%p\n",csr_ja_rows);
	delete(csr_ja_rows);	
	//printf("csc_a=%p\n",csc_a);
	delete(csc_a);	
	//printf("csc_ia=%p\n",csc_ia);
	delete(csc_ia);	
	//printf("csc_ja=%p\n",csc_ja);
	delete(csc_ja);	
	//printf("u=%p\n",u);
	delete(u);
	//printf("v=%p\n",v);
	delete(v);
	//printf("l_u=%p\n",l_u);
	delete(l_u);
	//printf("l_v=%p\n",l_v);
	delete(l_v);
	//printf("l_ub=%p\n",l_ub);
	delete(l_ub);
	//printf("l_vb=%p\n",l_vb);
	delete(l_vb);
	//printf("lse=%p\n",lse);
	delete(lse);
	//printf("dist_matrix=%p\n",dist_matrix);
	delete(dist_matrix);
	//printf("dist_topK=%p\n",dist_topK);
	delete(dist_topK);
	//printf("recommendations=%p\n",recommendations);
	delete(recommendations);
}

bool umatrix::init()
{
	cout<<"umatrix::init() called"<<endl;
	if (alloc() == false) {
		return false;
	}
	
	int curr_id = 0;
	for (int i=0; i<data_rwise.size(); i++) {
		for (int j=0; j<data_rwise[i].size(); j++) {
			int row = i, col = data_rwise[i][j].first;
			double rating = data_rwise[i][j].second;
			csr_a[curr_id] = rating;
			csr_ia[row+1] += 1;
			csr_ja[curr_id] = col;
			csr_ja_rows[curr_id] = row;
			curr_id++;
		}
	}
	cout<<"umatrix csr preparation done"<<endl;

	curr_id = 0;
	for (int i=0; i<data_cwise.size(); i++) {
		for (int j=0; j<data_cwise[i].size(); j++) {
			int col = i, row = data_cwise[i][j].first;
			double rating = data_cwise[i][j].second;
			csc_a[curr_id] = rating;
			csc_ia[col+1] += 1;
			csc_ja[curr_id] = row;
			curr_id++;
		}
	}
	cout<<"umatrix csc  preparation done"<<endl;

	for (int i=1; i<=rows; i++) {
		if (csr_ia[i] == 0) {
			cout<<"User-"<<i<<" hasn't rated anything"<<endl;
		}
		csr_ia[i]+= csr_ia[i-1];
	}

	for (int j=1; j<=cols; j++) {
		if (csc_ia[j] == 0) {
			cout<<"Item-"<<j<<" hasn't been rated anytime"<<endl;
		}
		csc_ia[j]+= csc_ia[j-1];
	}

	cout<<"umatrix::init() ended"<<endl;
	return true;
}

void umatrix::free_device()
{
	DEV_FREE(d_csr_a,"d_csr_a");
	DEV_FREE(d_csr_ia,"d_csr_ia");
	DEV_FREE(d_csr_ja,"d_csr_ja");
	DEV_FREE(d_csr_ja_rows,"d_csr_ja_rows");
	DEV_FREE(d_csc_a,"d_csc_a");
	DEV_FREE(d_csc_ia,"d_csc_ia");
	DEV_FREE(d_csc_ja,"d_csc_ja");
	DEV_FREE(d_u,"d_u");
	DEV_FREE(d_v,"d_v");
	DEV_FREE(d_l_u,"d_l_u");
	DEV_FREE(d_l_v,"d_l_v");
	DEV_FREE(d_l_ub,"d_l_ub");
	DEV_FREE(d_l_vb,"d_l_vb");
	DEV_FREE(d_lse,"d_lse");
	DEV_FREE(d_dist_matrix,"d_dist_matrix");
	DEV_FREE(d_dist_topK,"d_dist_topK");
	DEV_FREE(d_recommendations,"d_recommendations");
}

bool umatrix::alloc_device()
{
	cout<<"umatrix::alloc_device() called"<<endl;

	cout<<sizeof(double)*num_elems<<endl;
	if (!DEV_ALLOC((void **)&d_csr_a,sizeof(double)*num_elems,"d_csr_a")) goto ErrorHandler;
	cout<<sizeof(int)*(rows+1)<<endl;
	if (!DEV_ALLOC((void **)&d_csr_ia,sizeof(int)*(rows+1),"d_csr_ia")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_csr_ja,sizeof(int)*num_elems,"d_csr_ja")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_csr_ja_rows,sizeof(int)*num_elems,"d_csr_ja_rows")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_csc_a,sizeof(double)*num_elems,"d_csc_a")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_csc_ia,sizeof(int)*(cols+1),"d_csc_ia")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_csc_ja,sizeof(int)*num_elems,"d_csc_ja")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_u,sizeof(double)*(L*rows),"d_u")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_v,sizeof(double)*(L*cols),"d_v")) goto ErrorHandler;

	if (!DEV_ALLOC((void **)&d_l_u,sizeof(double)*(L*L*rows),"d_l_u")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_l_v,sizeof(double)*(L*L*cols),"d_l_v")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_l_ub,sizeof(double)*(L*rows),"d_l_ub")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_l_vb,sizeof(double)*(L*cols),"d_l_vb")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_lse,sizeof(double)*(num_elems),"d_lse")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_dist_matrix,sizeof(double)*(cols)*(cols-1)/2,"d_dist_matrix")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_dist_topK,sizeof(int)*(cols*RECOMMEND_K),"d_dist_topK")) goto ErrorHandler;
	if (!DEV_ALLOC((void **)&d_recommendations,sizeof(int)*(rows*RECOMMEND_K),"d_recommendations")) goto ErrorHandler;


	if (!TO_DEVICE(d_csr_a,csr_a,sizeof(double)*num_elems,"d_csr_a")) goto ErrorHandler;
	if (!TO_DEVICE(d_csr_ia,csr_ia,sizeof(int)*(rows+1),"d_csr_ia")) goto ErrorHandler;
	if (!TO_DEVICE(d_csr_ja,csr_ja,sizeof(int)*num_elems,"d_csr_ja")) goto ErrorHandler;
	if (!TO_DEVICE(d_csr_ja_rows,csr_ja_rows,sizeof(int)*num_elems,"d_csr_ja_rows")) goto ErrorHandler;
	if (!TO_DEVICE(d_csc_a,csc_a,sizeof(double)*num_elems,"d_csc_a")) goto ErrorHandler;
	if (!TO_DEVICE(d_csc_ia,csc_ia,sizeof(int)*(cols+1),"d_csc_ia")) goto ErrorHandler;
	if (!TO_DEVICE(d_csc_ja,csc_ja,sizeof(int)*num_elems,"d_csc_ja")) goto ErrorHandler;
	if (!TO_DEVICE(d_u,u,sizeof(double)*(L*rows),"d_u")) goto ErrorHandler;
	if (!TO_DEVICE(d_v,v,sizeof(double)*(L*cols),"d_v")) goto ErrorHandler;

	if (!TO_DEVICE(d_l_u,l_u,sizeof(double)*(L*L*rows),"d_l_u")) goto ErrorHandler;
	if (!TO_DEVICE(d_l_v,l_v,sizeof(double)*(L*L*cols),"d_l_v")) goto ErrorHandler;
	if (!TO_DEVICE(d_l_ub,l_ub,sizeof(double)*(L*rows),"d_l_ub")) goto ErrorHandler;
	if (!TO_DEVICE(d_l_vb,l_vb,sizeof(double)*(L*cols),"d_l_vb")) goto ErrorHandler;
	if (!TO_DEVICE(d_lse,lse,sizeof(double)*(num_elems),"d_lse")) goto ErrorHandler;
	if (!TO_DEVICE(d_dist_matrix,dist_matrix,sizeof(double)*(cols)*(cols-1)/2,"d_dist_matrix")) goto ErrorHandler;
	if (!TO_DEVICE(d_dist_topK,dist_topK,sizeof(int)*(cols*RECOMMEND_K),"d_dist_topK")) goto ErrorHandler;
	if (!TO_DEVICE(d_recommendations,recommendations,sizeof(int)*(rows*RECOMMEND_K),"d_recommendations")) goto ErrorHandler;

	cout<<"umatrix::alloc_device alloc succedded"<<endl;
	return true;
ErrorHandler:
	free_device();
	return false;
}

void umatrix::reload_uv()
{
	memset(u,0,sizeof(double)*(L*rows));
	if (!FROM_DEVICE(d_u,u,sizeof(double)*(L*rows),"u_data")) {
		cout<<"From device to host u-transfer failed"<<endl;
	}

	memset(v,0,sizeof(double)*(L*cols));
	if (!FROM_DEVICE(d_v,v,sizeof(double)*(L*cols),"v_data")) {
		cout<<"From device to host v-transfer failed"<<endl;
	}
}
