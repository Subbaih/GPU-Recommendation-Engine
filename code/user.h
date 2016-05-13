#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<utility>
#include<map>
#include<fstream>
#include<sstream>
#include<set>
using namespace std;

class user {
	public:
		user(): id(0), num_ratings(0) {};
		user(int uid): id(uid), num_ratings(0) {
		}
		
		user(string uid):num_ratings(0) {
			istringstream iss(uid.c_str());
			iss >> id;
		}

		int get_id() {
			return id;
		}

		void set_num_ratings(int i) {
			num_ratings = i;
		}

		int get_num_ratings() {
			return num_ratings;
		}
			
		void set_avg_rating(double avg) {
			avg_rating = avg;
		}

		double get_avg_rating() {
			return avg_rating;
		}

		void add_rating(int item_id, double rating) {
			ratings.push_back(make_pair<int,double>(item_id,rating));
			if (num_ratings == 0) {
				avg_rating = rating;
				num_ratings++;
			} else { 
				avg_rating = (avg_rating*num_ratings + rating);
				num_ratings = num_ratings + 1;
				avg_rating = avg_rating/num_ratings;
			}
		}

	private:
		int id;
		double avg_rating;
		int num_ratings;
		vector<pair<int,double> >ratings; // Item-Id and Ratings;
};
