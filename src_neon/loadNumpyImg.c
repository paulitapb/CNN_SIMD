#ifndef __LOADNUMPYIMG__
#define __LOADNUMPYIMG__

#include "main.h"

void load_image(generic_images images, FILE* image_ptr, long int* image_idx){

	// Given an .img file
	// Fill the generic images struct

	int c_img = read_int(image_ptr, image_idx);
	int img_size = read_int(image_ptr, image_idx);

	*image_idx += 2;

	for(int img = 0; img < c_img; img++){

		generic_image g_img = malloc(img_size * sizeof(generic_row));

		for(int i = 0; i < img_size; i++){

			generic_row r = malloc(img_size * sizeof(float));

			for(int j = 0; j < img_size; j++){

				r[j] = read_float(image_ptr, image_idx);
			}

			g_img[i] = r;

			*image_idx += 1;
		}

		images[img] = g_img;

		*image_idx += 1;
	}

}

void load_vect(float* vect, FILE* image_ptr, long int* image_idx){

	// Given an .txt file
	// Fill a vector

	int vect_size = read_int(image_ptr, image_idx);
	
	*image_idx += 2;
	for (int i = 0; i < vect_size; i++)
	{
		vect[i] = read_float(image_ptr, image_idx); 
	}
	
}

#endif