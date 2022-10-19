#ifndef __MAXPOOL__
#define __MAXPOOL__

#include "main.h"

float max_group(float* group, int len){

    // Given an array of floats and its lenght
    // Returns the maximun of the array

    float max = group[0];
	float* max_vect = malloc(2*sizeof(float));
    max_vect[0] = group[0];
    max_vect[1] = group[0];
    float32x2_t max_vect_neon = vld1_f32(max_vect);
    float local_max;
    int rest = len%4; 
    for (int i = 0; i < (len-rest); i+=4)
    { 
        //load 4 floats 
        float32x2_t group_1 = vld1_f32(&(group[i]));
        float32x2_t group_2 = vld1_f32(&(group[i+2]));

        // calculate max
        float32x2_t group_max = vmax_f32(group_1, group_2);
        group_max = vpmax_f32(group_max, group_max);
        max_vect_neon = vmax_f32(max_vect_neon, group_max);
    }
    
    max = vget_lane_f32(max_vect_neon, 0);

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
    float* group = malloc(pool_size * pool_size * sizeof(float));

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

    free(group);
    free_generic_images(input_images, n_images_in, image_in_size);

    return output_images;
}

#endif
