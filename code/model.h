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

#include "user.h"
#include "item.h"
#include "umatrix.h"

using namespace std;

class model 
{
	public:
		model(string f, int num_latents)
		{
			int ratings;
			L = num_latents;
			if (*f.rbegin()!='/')
				f = f + '/';
			fpath = f;
			string fname = fpath + "u.info";
			ifstream infile(fname.c_str());
			string line1, line2, line3, line4, s_usr, s_item, s_delim, s_ratings;
			getline(infile,line1); istringstream iss1(line1.c_str()); iss1>>num_usrs>>s_usr;
			getline(infile,line2); istringstream iss2(line2.c_str()); iss2>>num_items>>s_item;
			getline(infile,line3); istringstream iss3(line3.c_str()); iss3>>ratings>>s_ratings;
			getline(infile,line4); istringstream iss4(line4.c_str()); iss4>>delimiter>>s_delim;
			cout<<"Users="<<num_usrs<<" Items="<<num_items<<" Delimiter="<<delimiter<<endl;
		}

		void load_users();
		void load_items();
		void load_umatrix(); 
		bool init();
		void uninit();
		void debug();
		void print_uv();
		inline void reload_uv() {um.reload_uv();}; 
		inline void run_modeling(int max_iterations, double lambda) 
		{
			um.run_modeling(max_iterations,lambda);
		}
		inline void recommend() {um.run_recommendation();}

	private:
		string fpath;
		char delimiter;
		int num_usrs;
		int num_items;
		int L; 	// Latent Factors
		vector<user> users;
		vector<item> items;
		umatrix um;
};
