#ifndef __MAXPOOL__
#define __MAXPOOL__

#include "main.h"

float max_group(float* group, int len){

    // Given an array of floats and its lenght
    // Returns the maximun of the array

	float max = group[0];
    int rest = len%8; 
    for (int i = 0; i < (len-rest); i+=8)
    { 
        //load 8 floats 
        __m128 v1 = _mm_load_ps(&group[i]); 
        __m128 v2 = _mm_load_ps(&group[i+4]);
        
        __m128 max_v = _mm_max_ps(v1, v2);  // get max values 
        
        __m128 shuf = _mm_movehdup_ps(max_v); //duplicate odd pos  

        max_v = _mm_max_ps(max_v, shuf); //get max values 

        shuf = _mm_movehl_ps(max_v, shuf); // move high half to lower half 

        max_v = _mm_max_ss(max_v, shuf); //max lower pos 

        __attribute__ ((aligned (32))) float m = _mm_cvtss_f32(max_v); //cast lower pos as float
        if(m > max){
            max = m;
        } 
    }
    //process the rest 
	for(int i = 0; i < rest; i++){
        
		if(max < group[(len-rest) + i]){

			max = group[(len-rest) + i];
		}
        
	}

	return max;
}

generic_images maxpool(generic_images input_images, maxpool_struct *parameters){

    // Given a generic image and a maxpool structure
    // Returns the maxpool of the input generic image

	// read parameters
    int n_images_in = parameters->dim_in[0];
    int image_in_size = parameters->dim_in[1];

    int pool_size = parameters->pool_size;
    int strides = parameters->strides;

    int n_images_out = parameters->dim_out[0];
    int image_out_size = parameters->dim_out[1];

    // Create output image
    generic_images output_images = create_images(n_images_out, image_out_size, 0);

    // variable for saving groups of values where to take the maximun
    float* group = _mm_malloc(pool_size * pool_size * sizeof(float), 32);

    // For each input / output image
    for(int img_in = 0; img_in < n_images_in; img_in++){

        // For each image section by row where to apply maxpool
    	for(int i = 0; i < image_in_size - pool_size + 1; i += strides){

            // For each image section by column where to apply maxpool
    		for(int j = 0; j < image_in_size - pool_size + 1; j += strides){

                // For each row in pool_size
    			for(int p = 0; p < pool_size; p++){

                    // For each column in pool_size
    				for(int q = 0; q < pool_size; q++){

                        // Assign value to group
    					group[(p*pool_size) + q] = input_images[img_in][i+p][j+q];

    				}
    			}

                // Assing maximun in each group to output
    			output_images[img_in][(int)(i/strides)][(int)(j/strides)] = max_group(group, pool_size  * pool_size);
    		}
    	}
    }

    _mm_free(group);
    free_generic_images(input_images, n_images_in, image_in_size);

    return output_images;
}

#endif