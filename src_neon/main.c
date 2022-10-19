#include "main.h"


int main(void){

	
	dataset imgs = malloc(sizeof(generic_images)*test_size);
	unsigned int* labels = malloc(sizeof(unsigned int)*test_size);  
	loadData(labels, imgs); 

	// Read file
    FILE* file_ptr = fopen("../modelos_redes/arm_model/arm_model.txt", "r"); 
    
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
		free(res);

	}
	
	

	free_data_set(imgs, n_used, test_size, 1, height);
	free_net(loaded_net);
	free(file_idx);
	free(labels);

	return 0;
}
