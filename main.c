#include "main.h"
#include <x86intrin.h> //all instrinsics

int main(void){
        
	//===== TEST =====
	/*
	test_execution();
	test_load_layers_all();
	test_completos();
	load_net_test_completo();
	execute_net_test_completo();
	*/
	
	dataset imgs = malloc(sizeof(generic_images)*test_size);
	unsigned int* labels = malloc(sizeof(unsigned int)*test_size);  
	loadData(labels, imgs); 

	// Read file
    FILE* file_ptr = fopen("../modelos_redes/intel_avx_model/intel_avx_model.txt", "r"); 
    
    // Check file_ptr
    if(file_ptr == NULL){
        printf("Error al cargar la red");  
        exit(-1); 
    }
    //printf("red file cargada \n");
    
    // Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;

	net* loaded_net = malloc(sizeof(net)); 
	loadNet(loaded_net, file_ptr, file_idx);
	
	int n_used = test_size;

	float* res; //execute net

	for(int i = 0; i < n_used; i++){

		res = executeNet(loaded_net, imgs[i]);
		//print_vect(res,10);
		_mm_free(res);

	}

	free_data_set(imgs, n_used, test_size, 1, height);
	free_net(loaded_net);
	free(file_idx);
	free(labels);
	return 0;
}
