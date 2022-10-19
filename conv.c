#ifndef __CONV__
#define __CONV__

#include "main.h"

generic_images conv(generic_images input_images, conv_struct *parameters){

    // Given a generic image and a conv_struct structure
    // Returns the convolution of the input generic image

    // read parameters
    int n_images_in = parameters->dim_in[0];
    int image_in_size = parameters->dim_in[1];

    int kernel_size = parameters->kernel_size;
    int strides = parameters->strides;

    int n_images_out = parameters->dim_out[0];
    int image_out_size = parameters->dim_out[1];

    // Create output image
    generic_images output_images = create_images(n_images_out, image_out_size, 0);

    // Create auxiliar variables
    conv_kernel kernel;
    int kernel_vect_len;
    int rem;
    float sum;

    // For each output image
    for(int img_out = 0; img_out < n_images_out; img_out++){

        // For each input image
        for(int img_in = 0; img_in < n_images_in; img_in++){

            // Assign kernel
            kernel = parameters->kernels[img_in][img_out];

            // vectorize and align kernel in memory
            __attribute__ ((aligned (32))) float kernel_vect[(kernel_size*kernel_size)];

            for(int i = 0; i < kernel_size; i++){

                for(int j = 0; j < kernel_size; j++){

                    kernel_vect[(i*kernel_size) + j] = kernel[i][j];
                }
            }

            // calculate kernel vect dimensions
            rem = (kernel_size*kernel_size)%8;
            kernel_vect_len = (kernel_size*kernel_size);

            // For each kernel position image by row
            for(int i = 0; i < image_in_size - kernel_size + 1; i += strides){

                // For each kernel position image by column
                for(int j = 0; j < image_in_size - kernel_size + 1; j += strides){

                    // vectorize and align image section in memory
                    __attribute__ ((aligned (32))) float img_section[(kernel_size*kernel_size)];

                    for(int img_row = 0; img_row < kernel_size; img_row++){

                        for(int img_col = 0; img_col < kernel_size; img_col++){

                            img_section[(img_row*kernel_size) + img_col] = input_images[img_in][i+img_row][j+img_col];
                        }
                    }

                    // "sum" of kernel multiplication
                    // is the result of applying a kernel in one image section

                    sum = 0;
                    
                    for(int vect_idx = 0; vect_idx < kernel_vect_len-rem; vect_idx+=8){

                        // load __m256
                        __m256 kernel_vect_m256 = _mm256_load_ps(&(kernel_vect[vect_idx]));
                        __m256 img_section_m256 = _mm256_load_ps(&(img_section[vect_idx]));

                        // multiply
                        __m256 mul = _mm256_mul_ps(kernel_vect_m256, img_section_m256); 

                        // horizontal add
                        __m128 vlow  = _mm256_castps256_ps128(mul); //cast lower pos as _m128 
                        __m128 vhigh = _mm256_extractf128_ps(mul, 1); // high 128
                        vlow  = _mm_add_ps(vlow, vhigh);     // add low and high -> 4 partial sums 
                        __m128 shuf = _mm_movehdup_ps(vlow); //duplicate odd pos 
                        __m128 sums = _mm_add_ps(vlow, shuf); //partials sums in even pos 
                        shuf = _mm_movehl_ps(shuf, sums); // move partial sum to lower pos 
                        sums = _mm_add_ss(sums, shuf); //add lower pos 
                        sum += _mm_cvtss_f32(sums); //cast lower pos as float
                    }

                    // multiply and add non-vectorized values
                    for(int idx = kernel_vect_len - rem; idx < kernel_vect_len; idx++){

                        sum += kernel_vect[idx] * img_section[idx];
                    }

                    // Assign sum to output image
                    output_images[img_out][(int)i/strides][(int)j/strides] += sum;
                }
            }
        }
    }

    int rem_out = image_out_size % 8;

    // For each output image
    for(int img_out = 0; img_out < n_images_out; img_out++){

        // For each row in output image
        for(int i = 0; i < image_out_size; i++){

            // For each column in output image
            for(int j = 0; j < image_out_size - rem_out; j+=8){

                // Add bias
                //output_images[img_out][i][j] += parameters->bias[img_out];

                __m256 out_img = _mm256_load_ps(&(output_images[img_out][i][j]));

                __m256 bias_m256 = _mm256_set1_ps(parameters->bias[img_out]);

                __m256 sum_bias = _mm256_add_ps(out_img, bias_m256);
                
                // Apply relu activation function
                __m256 v2 = _mm256_setzero_ps();  
                __m256 max = _mm256_max_ps(sum_bias, v2);  
                _mm256_store_ps(&(output_images[img_out][i][j]), max);

                

            }

            for(int j = image_out_size - rem_out; j < image_out_size; j++){

                sum = output_images[img_out][i][j] + parameters->bias[img_out];
                output_images[img_out][i][j] = relu(sum);
            }
        }
    }

    free_generic_images(input_images, n_images_in, image_in_size);

    return output_images;
}

#endif