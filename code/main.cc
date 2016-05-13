#include "model.h"

int num_latents;
int main(int argn, char *argv[])
{
	if (argn != 5) {
		printf("./main $data_location_index $num_latents $num_iterations $lambda\n");
		return 0;
	}
	
	int ds_idx = atoi(argv[1]);
	if (ds_idx <0 || ds_idx > 3) {
		printf("Illegal data_location_index: range is 0 to 3\n");
		return 0;
	}
	num_latents = atoi(argv[2]);
	int max_iterations = atoi(argv[3]);
	double lambda = atof(argv[4]);
	
	string dpath[] = {"../data/ml-100k","../data/ml-1m","../data/ml-10m","../data/ml-6"};
	model m(dpath[ds_idx],num_latents);
	m.init();
	
	/*	
	m.debug();
	m.reload_uv();
	cout<<"After uv reload"<<endl;
	m.debug();
	*/
	
	//m.run_modeling(max_iterations,lambda);
	m.recommend();
	m.uninit();
	return 0;
}
