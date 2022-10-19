#ifndef __FLATTEN__
#define __FLATTEN__

#include "main.h"

float* flatten(generic_images input_images, flatten_struct* flatten){
    
    // Given a generic_image input_image
    // Return a vectorization of input_image:
    // first by image, then by row, then by column

    float* res = malloc(sizeof(float)*(flatten->dim_out)); // return vector

    // for each row
    for (int i = 0; i < (flatten->dim_in[2]); i++) 
    {
        // for each column
        for (int j = 0; j < (flatten->dim_in[1]); j++)
        {
            // for each image
            for (int k = 0; k < (flatten->dim_in[0]); k++)
            {
                // Assign value to corresponding res place
                res[(i*(flatten->dim_in[0])*(flatten->dim_in[1])) + 
                (j* (flatten->dim_in[0])) + k] = input_images[k][i][j];  
                 
            }
            
        }
        
    }

    int n_images_in = flatten->dim_in[0];
    int image_in_size = flatten->dim_in[1];

    free_generic_images(input_images, n_images_in, image_in_size);

    return res; 
}

#endif