#ifndef __LOADNET__
#define __LOADNET__

#include "main.h"

// function read int
int read_int(FILE *file_ptr, long int *file_idx){

    // Given a file pointer and an index
    // return the integer value written in the file at the position index

    // Calculate integers longitud
    char* aux = malloc(sizeof(char)); //auxiliar variables
    *aux = '0';

    long int offset = 0; // size of the integer to read +1

    fseek(file_ptr, *file_idx, SEEK_SET); // set index on file

    //while there is no tab
    while(*aux != '\t'){

        fread(aux, sizeof(char), 1, file_ptr); //writes char in aux
        offset++;
    }
    
    free(aux);

    long int size_to_read = offset - 1; // calculate size of the integer to read

    char* n_char = malloc((size_to_read+1) * sizeof(char)); //string where to load the integer

    fseek(file_ptr, *file_idx, SEEK_SET); // set index
    fread(n_char, sizeof(char), size_to_read, file_ptr); // writes n_char

    n_char[size_to_read] = '\0';

    int n = atoi(n_char); // cast string to int

    free(n_char);

    *file_idx += offset; // update idx

    return n;
}

// function read float
float read_float(FILE *file_ptr, long int *file_idx){

    // Given a file pointer and an index
    // return the float value written in the file at the position index
    // same code as read_int but using atof insteaf of atoi

    char* aux = malloc(sizeof(char));
    *aux = '0';

    long int offset = 0;

    fseek(file_ptr, *file_idx, SEEK_SET);

    while(*aux != '\t'){

        fread(aux, sizeof(char), 1, file_ptr);
        offset++;
    }
    
    free(aux);

    long int size_to_read = offset - 1;

    char* n_char = malloc((size_to_read+1) * sizeof(char));

    fseek(file_ptr, *file_idx, SEEK_SET);
    fread(n_char, sizeof(char), size_to_read, file_ptr); 

    n_char[size_to_read] = '\0';

    float n = atof(n_char);

    free(n_char);

    *file_idx += offset;

    return n;
}

// function read string
char* read_string(FILE *file_ptr, long int *file_idx){

    // Given a file pointer and an index
    // return the string value written in the file at the position index
    // same code as read_int but with no cast to interger

    char* aux = malloc(sizeof(char));
    *aux = '0';

    long int offset = 0;

    fseek(file_ptr, *file_idx, SEEK_SET);

    while(*aux != '\t'){

        fread(aux, sizeof(char), 1, file_ptr);
        offset++;
    }
    
    free(aux);

    long int size_to_read = offset - 1;

    char* string = malloc(size_to_read * sizeof(char));

    fseek(file_ptr, *file_idx, SEEK_SET);
    fread(string, sizeof(char), size_to_read, file_ptr);

    *file_idx += offset;

    return string;
}

conv_kernels read_conv_kernels(FILE* file_ptr, long int *file_idx, int dim_in, int dim_out, int kernel_size){

    // Create kernel
	conv_kernels kernels = malloc(dim_in * sizeof(conv_kernels_row));

	// For each image input
	for(int img_in = 0; img_in < dim_in; img_in++){

		// Create kernels row
		conv_kernels_row kernels_row = malloc(dim_out * sizeof(conv_kernel));

		// For each image output
		for(int img_out = 0; img_out < dim_out; img_out++){

			// Create kernel for image_in image_out
			conv_kernel kernel = malloc(kernel_size * sizeof(conv_kernels_row));

			// For each kernel row
			for(int i = 0; i < kernel_size; i++){

				// Create row
				conv_kernel_row kernel_row = malloc(kernel_size * sizeof(float));

				// For each kernel column
				for(int j = 0; j < kernel_size; j++){

					// Assign value to kernel in row column
					kernel_row[j] = read_float(file_ptr, file_idx);
				}

				// Assing kernel row in kernel
				kernel[i] = kernel_row;

                *file_idx+=1; // Next line content
			}

			// Assign kernel in kernels row
			kernels_row[img_out] = kernel;

            *file_idx += 1; //Next kernel content
		}

		// Assign kernels_row in kernels
		kernels[img_in] = kernels_row;
	}

    return kernels;
}

conv_bias read_conv_bias(FILE* file_ptr, long int *file_idx, int dim_out){

    // Create bias
	conv_bias bias = malloc(dim_out * sizeof(float));

	for(int img_out = 0; img_out < dim_out; img_out++){

		bias[img_out] = read_float(file_ptr, file_idx);
	}

    *file_idx += 2;

    return bias;
}

void load_conv(conv_struct *conv_layer, FILE* file_ptr, long int *file_idx){

    // Read dim_in
    conv_layer->dim_in[0] = read_int(file_ptr, file_idx);
    conv_layer->dim_in[1] = read_int(file_ptr, file_idx);
    read_int(file_ptr, file_idx);

    // Read dim_out
    conv_layer->dim_out[0] = read_int(file_ptr, file_idx);
    conv_layer->dim_out[1] = read_int(file_ptr, file_idx);
    read_int(file_ptr, file_idx);

    // Read kernel size and strides
    conv_layer->kernel_size = read_int(file_ptr, file_idx);
    conv_layer->strides = read_int(file_ptr, file_idx);

    *file_idx += 2;

    // Read kernel and bias
    conv_layer->kernels = read_conv_kernels(file_ptr, file_idx, 
                                            conv_layer->dim_in[0], 
                                            conv_layer->dim_out[0], 
                                            conv_layer->kernel_size);

    conv_layer->bias = read_conv_bias(file_ptr, file_idx, conv_layer->dim_out[0]);
}

void load_maxpool(maxpool_struct *mp_layer, FILE *file_ptr, long int *file_idx){

    // Read input dimension
    mp_layer->dim_in[0] = read_int(file_ptr, file_idx);
    mp_layer->dim_in[1] = read_int(file_ptr, file_idx);
    mp_layer->dim_in[2] = read_int(file_ptr, file_idx);

    // Read output dimension
    mp_layer->dim_out[0] = read_int(file_ptr, file_idx);
    mp_layer->dim_out[1] = read_int(file_ptr, file_idx);
    mp_layer->dim_out[2] = read_int(file_ptr, file_idx);

    // Read pool size and strides
    mp_layer->pool_size = read_int(file_ptr, file_idx);
    mp_layer->strides = read_int(file_ptr, file_idx);

    *file_idx += 2;
}

void load_flatten(flatten_struct *fl_layer, FILE *file_ptr, long int *file_idx){
    
    // Read input dimension
    fl_layer->dim_in[0] = read_int(file_ptr, file_idx);  
    fl_layer->dim_in[1] = read_int(file_ptr, file_idx);  
    fl_layer->dim_in[2] = read_int(file_ptr, file_idx);

    // Read output dimension
    fl_layer->dim_out= read_int(file_ptr, file_idx);

    *file_idx += 2;
    
}

void load_dense(dense_struct *dense_layer, FILE *file_ptr, long int *file_idx){

    // Read input and output dimension
    dense_layer->dim_in = read_int(file_ptr, file_idx);
    dense_layer->dim_out = read_int(file_ptr, file_idx);

    // Read activation function
    char* act_func = read_string(file_ptr, file_idx);

    for(int i = 0; i < 4; i++)
    {
        dense_layer->actFunc[i] = act_func[i];
    }

    free(act_func);

    *file_idx += 2;

    //load weights
    dense_kernel weights = malloc(sizeof(dense_kernel_row)*(dense_layer->dim_out)); 
    for (int i = 0; i < (dense_layer->dim_out); i++)
    {
        dense_kernel_row r = malloc(sizeof(float)*(dense_layer->dim_in)); 
        for (int j = 0; j < dense_layer->dim_in; j++)
        {
            r[j] = read_float(file_ptr,  file_idx);
        }
        weights[i] = r;
        *file_idx += 1;
    }
    *file_idx += 1;

    // load bias 
    dense_bias bias = malloc(sizeof(float)*(dense_layer->dim_out)); 
    for (int i = 0; i < dense_layer->dim_out; i++)
    {
        bias[i] = read_float(file_ptr,  file_idx);  
    }

    *file_idx += 2;

    dense_layer->weights = weights;
    dense_layer->bias = bias;
}

void loadNet(net* loaded_net, FILE* file_ptr, long int* file_idx){

    // Given a net struct and a path to .orga file as a string
    // Returns the net struct filled with the file parameters

    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

    //allocate net struct 
    loaded_net->cant_capas = n_layers;
    loaded_net->arq_names = malloc(sizeof(char)*(n_layers)); //allocate array of names 
    loaded_net->arq_structs = malloc(sizeof(void*)*(n_layers)); //allocate array of pointer to structs 
    
    *file_idx += 2;

    char* title; // layer title variable

    uint8_t layer_counter = 0;

    // Read file by layers
    while(layer_counter < n_layers){

        title = read_string(file_ptr, file_idx); // read title
         
        switch(title[0]){ // C: convolution, M: maxpool, F: flatten, D: dense

            case 'C': // convolution layer case

                loaded_net->arq_names[layer_counter] = 'C'; // Append layer name

                // Assign conv struct pointer in net
                conv_struct* conv_layer = malloc(sizeof(conv_struct)); // reserves memory
                load_conv(conv_layer, file_ptr, file_idx); // load layer
                loaded_net->arq_structs[layer_counter] = conv_layer; // load layer in net struct

                //printf("conv cargada :D\n");
                break;

            case 'M': // maxpooling layer case

                loaded_net->arq_names[layer_counter] = 'M'; // Append layer name

                maxpool_struct* mp_layer = malloc(sizeof(maxpool_struct)); // reserves memory
                load_maxpool(mp_layer, file_ptr, file_idx); // load layer
                loaded_net->arq_structs[layer_counter] = mp_layer; // load layer in net struct

                //printf("maxpool cargada :D\n");
                break;

            case 'F': // flatten layer case

                loaded_net->arq_names[layer_counter] = 'F'; // Append layer name

                flatten_struct* fl_layer = malloc(sizeof(flatten_struct)); // reserves memory
                load_flatten(fl_layer, file_ptr, file_idx); // load layer
                loaded_net->arq_structs[layer_counter] = fl_layer; // load layer in net struct

                //printf("flatten cargada :D\n");
                break;

            case 'D': // dense layer case

                loaded_net->arq_names[layer_counter] = 'D'; // Append layer name

                dense_struct* dense_layer = malloc(sizeof(dense_struct)); // reserves memory
                load_dense(dense_layer, file_ptr, file_idx); // load layer
                loaded_net->arq_structs[layer_counter] = dense_layer; // load layer in net struct

                //printf("dense cargada :D\n");
                break;

            default: // not a recognazible layer

                printf("Error al cargar la net\n"); 
                break; 
        }

        layer_counter++;
        free(title);
    }
 
}
#endif
