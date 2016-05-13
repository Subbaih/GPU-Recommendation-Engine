#include "model.h"
#include<ctime>

void model::load_users()
{
	users = vector<user>(num_usrs);
	string fname = fpath + "u.user";
	string line;
	ifstream infile(fname.c_str());
	while (getline(infile,line)) {
		istringstream iss(line.c_str());
		string s_id;
		getline(iss,s_id,delimiter);		
		istringstream iss2(s_id.c_str());
		int id;
		iss2>>id;
		users[id-1] = user(id);
	}
}

void model::load_items()
{
	items = vector<item>(num_items);
	string fname = fpath + "u.item";
	string line;
	ifstream infile(fname.c_str());
	while (getline(infile,line)) {
		istringstream iss(line.c_str());
		string s_id;
		getline(iss,s_id,delimiter);		
		istringstream iss2(s_id.c_str());
		int id;
		iss2>>id;
		items[id-1] = item(id);
	}
}

void model::load_umatrix()
{
	um.load_umatrix(num_usrs,num_items,fpath,L,delimiter);
	for (int i=0; i<num_usrs; i++) {
		//cout<<"User-"<<i<<" has "<<um.data_rwise[i].size()<<" ratings"<<endl;
		for (int j=0; j<um.data_rwise[i].size(); j++) {
			pair<int,double> p = um.data_rwise[i][j]; 
			int uid = i, iid = p.first;
			//cout<<"user="<<uid<<" iid="<<iid<<endl;
			users[uid].add_rating(iid,p.second);
			items[iid].add_rating(uid,p.second);
		}
	}
	/* Print utility matrix in nice format
	for (int i=0; i<num_usrs; i++) {
		for (int j=0; j<num_items; j++) {
			printf("%g ",um(i,j));
		}
		printf("\n");
	}
	*/
	cout<<"load_umatrix done"<<endl;
}

bool model::init()
{
	cout<<"Loading users"<<endl;
	load_users();
	cout<<"Loading items"<<endl;
	load_items();
	cout<<"Loading utility matrix"<<endl;
	load_umatrix();
	if (um.init() == false) { 
		return false;
	}
	cout<<"utility matrix inited"<<endl;

	srand(time(0));
	for (int i=0; i<num_usrs; i++) {
		if (users[i].get_num_ratings() == 0) {
			cout<<"Model: User "<<i<<" has no rating"<<endl;
		}
		um.set_u(i,users[i].get_avg_rating());	
	} 
	
	for (int i=0; i<num_items; i++) {
		if (items[i].get_num_ratings() == 0) { 
			cout<<"Model: Item "<<i<<" has no rating"<<endl;
		}
		um.set_v(i,items[i].get_avg_rating());	
	}

	if (um.alloc_device() == false) {
		um.um_free();
	}
	cout<<"model inited"<<endl;
	return true;
}

inline void print_vector(double *p, int dim, string suffix, int i)
{
	cout<<suffix<<"-"<<i<<" ";
	for (int i=0; i<dim; i++) {
		printf("%g ",p[i]);
	}
	printf("\n");
}

void model::print_uv()
{
	double *temp = new double[L];
	if (temp == NULL) {
		printf("Unable to allocate for print_uv temp\n");
	}

	for (int i=0; i<num_usrs; i++) {
		um.get_u(temp,i);
		print_vector(temp,L,"User:",i);	
	} 
	
	for (int i=0; i<num_items; i++) {
		um.get_v(temp,i);
		print_vector(temp,L,"Item:",i);	
	} 
	delete temp;
}

void model::debug()
{
	print_uv();
}

void model::uninit()
{
	um.free_device();
	um.um_free();
}
