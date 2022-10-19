#ifndef __PRINTS__
#define __PRINTS__

#include "main.h"

void print_conv_kernels(conv_struct* conv_layer){

	printf("Convolutional kernels\n\n");

	for(int c_in = 0; c_in < conv_layer->dim_in[0]; c_in++){

		for(int c_out = 0; c_out < conv_layer->dim_out[0]; c_out++){

			for(int i = 0; i < conv_layer->kernel_size; i++){

				for(int j = 0; j < conv_layer->kernel_size; j++){

					printf("%.10f\t", conv_layer->kernels[c_in][c_out][i][j]);
				}

				printf("\n");
			}

			printf("\n");
		}
	}
}

void print_conv_bias(conv_struct* conv_layer){

	printf("Convolutional bias\n\n");

	for(int i = 0; i < conv_layer->dim_out[0]; i++){

		printf("%.10f\t", conv_layer->bias[i]);
	}

	printf("\n");
}

void print_dense_kernel(dense_struct* dense_layer){

	printf("Dense kernel\n\n");

	for(int c_out = 0; c_out < dense_layer->dim_out; c_out++){

		for(int c_in = 0; c_in < dense_layer->dim_in; c_in++){

			printf("%.10f\t", dense_layer->weights[c_out][c_in]);
		}

		printf("\n");
	}

	printf("\n");
}

void print_dense_bias(dense_struct* dense_layer){

	printf("Dense bias\n\n");

	for(int i = 0; i < dense_layer->dim_out; i++){

		printf("%.10f\t", dense_layer->bias[i]);
	}

	printf("\n");
}

void print_images(generic_images res, int* dim){

	// Given a generic image and its dimesions
	// print the image

	for(int img = 0; img < dim[0]; img++){

		for(int i = 0; i < dim[1]; i++){

			for(int j = 0; j < dim[1]; j++){

				printf("%.10f\t", res[img][i][j]);
			}

			printf("\n");
		}

		printf("\n");
	}
}

void print_vect(float* res, uint8_t n){
	printf("[ "); 
	for (int i = 0; i < n; i++)
	{
		printf(" %.10f", res[i]); 
	}
	printf(" ]\n");
}

#endif