#ifndef __TEST__
#define __TEST__
#include "main.h"


void test_execution(){
	conv_test();
	maxpool_test();
	flatten_test();
	dense_relu_test(); 
	dense_soft_test();
}

void test_load_layers_all(){
	load_conv_test(); 
	load_flatten_test(); 
	load_maxpool_test();
	load_dense_relu_test();
	load_dense_softmax_test(); 
}

void test_completos(){
	conv_test_completo();
	maxpool_test_completo();
	flatten_test_completo();
	dense_relu_test_completo();
	dense_softmax_test_completo();
}

// layers tests

void conv_test(){

	printf("---------- conv test ----------\n");

	// Test convolutional layer

	int n_images_in = 2;
	int image_size = 5;

	int kernel_size = 3;

	int n_images_out = 2;

	// Create array of images
	generic_images images = create_images(n_images_in, image_size, 1);

	// Create conv struct
	conv_struct *conv_layer = malloc(sizeof(conv_struct));

	// dimension in
	conv_layer-> dim_in[0] = n_images_in;
	conv_layer-> dim_in[1] = image_size;
	conv_layer-> dim_in[2] = image_size;

	// dimension out
	conv_layer-> dim_out[0] = n_images_out;
	conv_layer-> dim_out[1] = image_size-3; // kernel 3x3 strides 2
	conv_layer-> dim_out[2] = image_size-3; // kernel 3x3 strides 2

	// kernel size
	conv_layer->kernel_size = 3;

	// Create kernel
	conv_kernels kernels = malloc( n_images_in * sizeof(conv_kernels_row));

	// For each image input
	for(int img_in = 0; img_in < n_images_in; img_in++){

		// Create kernels row
		conv_kernels_row kernels_row = malloc(n_images_out * sizeof(conv_kernel));

		// For each image output
		for(int img_out = 0; img_out < n_images_out; img_out++){

			// Create kernel for image_in image_out
			conv_kernel kernel = malloc(kernel_size * sizeof(conv_kernels_row));

			// For each kernel row
			for(int i = 0; i < kernel_size; i++){

				// Create row
				conv_kernel_row kernel_row = malloc(kernel_size * sizeof(float));

				// For each kernel column
				for(int j = 0; j < kernel_size; j++){

					// Assign value to kernel in row column
					kernel_row[j] = 1;
				}

				// Assing kernel row in kernel
				kernel[i] = kernel_row;
			}

			// Assign kernel in kernels row
			kernels_row[img_out] = kernel;
		}

		// Assign kernels_row in kernels
		kernels[img_in] = kernels_row;
	}

	// Create bias
	conv_bias bias = malloc(n_images_out * sizeof(float));

	for(int img_out = 0; img_out < n_images_out; img_out++){

		bias[img_out] = 1;
	}

	conv_layer->kernels = kernels; // Assign kernel
	conv_layer->bias = bias; // Assign bias

	conv_layer->strides = 2; // Assign strides

	// Execute convolutional layer
	generic_images images_out;

	images_out = conv(images, conv_layer);

	if(images_out[0][0][0] == 19 && images_out[1][0][0] == 19){
		printf("convolution ok!\n");
	}
	else{
		printf("convolution FAIL :(\n");
	}

}

void maxpool_test(){

	printf("---------- maxpool test ----------\n");

	// Test maxpool layer

	int n_images_in = 2;
	int image_size = 4;

	int n_images_out = n_images_in;

	// Create array of images
	generic_images images = create_images(n_images_in, image_size, 1);

	// Create conv struct
	maxpool_struct *maxpool_layer = malloc(sizeof(maxpool_struct));

	//strcpy(maxpool_layer->title,"MaxPooling2D");                                   // title

	// dimension in
	maxpool_layer-> dim_in[0] = n_images_in;
	maxpool_layer-> dim_in[1] = image_size;
	maxpool_layer-> dim_in[2] = image_size;

	// dimension out
	maxpool_layer-> dim_out[0] = n_images_out;
	maxpool_layer-> dim_out[1] = (int)(image_size/2); // pool_size 2 strides 2
	maxpool_layer-> dim_out[2] = (int)(image_size/2); // pool_size 2 strides 2

	maxpool_layer->pool_size = 2;// Assign pool size

	maxpool_layer->strides = 2; // Assign strides

	// Execute convolutional layer
	generic_images images_out;

	images_out = maxpool(images, maxpool_layer);

	if(images_out[0][0][0] == 1 && images_out[1][0][0] == 1){
		printf("maxpooling ok!\n");
	}
	else{
		printf("maxpooling FAIL :(\n");
	}

} 

void flatten_test(){

	printf("---------- flatten test ----------\n");

	int n_images_in = 2;
	int image_size = 4;

	// Create array of images
	generic_images images = create_images_dif_val_row(n_images_in, image_size); 

	// Create flatten struct
	flatten_struct *f = malloc(sizeof(flatten_struct)); 

	// Set input and output dimensions
	uint8_t d_in[3] = {2, 4, 4}; 
	f->dim_in[0] = d_in[0]; 
	f->dim_in[1] = d_in[1]; 
	f->dim_in[2] = d_in[2]; 
	f->dim_out = 32;  

	// Apply flatten layer
	float* flat = flatten(images, f); 

	// Check output
	uint8_t res = 1;

	for (int i = 0; i < 16; i++)
	{
		if (!((i/2 >= flat[i]) && (i/2 < flat[i] + 1))){
			res *= 0;
		}
	}

	if(res == 1){
		printf("flatten ok!\n");
	}
	else{
		printf("flatten FAIL :(\n");
	}
}

void dense_relu_test(){

	printf("---------- dense relu test ----------\n");

	// Create dense struct
	dense_struct *den = malloc(sizeof(dense_struct));  
	
	// Set input and output dimension
	den->dim_in = 5; 
	den->dim_out = 2;

	// Create bias
	dense_bias bias = malloc(den->dim_out * sizeof(float));
	
	bias[0] = 1;
	bias[1] = 1;

	// Create kernel
	dense_kernel wei = malloc(den->dim_out * sizeof(dense_kernel_row)); 
	
	for (int i = 0; i < den->dim_out; i++)
	{
		dense_kernel_row row = malloc(den->dim_in * sizeof(float));

		for(int j = 0; j < den->dim_in; j++)
		{
			row[j] = 1;
		}
		wei[i] = row;
	} 

	// Set bias and kernel
	den->bias = bias; 
	den->weights = wei;

	// Set activation function to relu
	den->actFunc[0] = 'r';
	den->actFunc[1] = 'e';
	den->actFunc[2] = 'l';
	den->actFunc[3] = 'u';

	// Create input vector for dense function
	float* x_in = malloc(den->dim_in * sizeof(float));

	for(int i = 0; i < den->dim_in; i++)
	{
		x_in[i] = 1;
	}

	// Apply dense layer
	float* d = dense(x_in, den); 


	// Check result
	if(d[0] == 6.0 && d[1] == 6.0)
	{
		printf("dense relu ok!\n");
	}
	else
	{
		printf("dense relu FAIL :(\n");
	}
}

void dense_soft_test(){

	printf("---------- dense soft test ----------\n");

	// Create dense struct
	dense_struct *den = malloc(sizeof(dense_struct));  
	
	// Set input and output dimension
	den->dim_in = 5; 
	den->dim_out = 2;

	// Create bias
	dense_bias bias = malloc(den->dim_out * sizeof(float));
	
	bias[0] = 1;
	bias[1] = 1;

	// Create kernel
	dense_kernel wei = malloc(den->dim_out * sizeof(dense_kernel_row)); 
	
	for (int i = 0; i < den->dim_out; i++)
	{
		dense_kernel_row row = malloc(den->dim_in * sizeof(float));

		for(int j = 0; j < den->dim_in; j++)
		{
			row[j] = 1;
		}
		wei[i] = row;
	} 

	// Set bias and kernel
	den->bias = bias; 
	den->weights = wei;

	// Set activation function to softmax
	den->actFunc[0] = 's';
	den->actFunc[1] = 'o';
	den->actFunc[2] = 'f';
	den->actFunc[3] = 't';

	// Create input vector for dense function
	float* x_in = malloc(den->dim_in * sizeof(float));

	for(int i = 0; i < den->dim_in; i++)
	{
		x_in[i] = 1;
	}

	// Apply dense layer
	float* d = dense(x_in, den); 

	// Check result
	if(d[0] == 0.5 && d[1] == 0.5)
	{
		printf("dense softmax ok!\n");
	}
	else
	{
		printf("dense softmax FAIL :(\n");
	}
}

// Load layers tests

void load_conv_test(){
	
	printf("---------- load conv test ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_conv_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	conv_struct* conv_layer = malloc(sizeof(conv_struct)); // reserves memory
    load_conv(conv_layer, file_ptr, file_idx); // load layer

	printf("dim_in: %d\t%d\n", conv_layer->dim_in[0], conv_layer->dim_in[1]);
	printf("dim_out: %d\t%d\n", conv_layer->dim_out[0], conv_layer->dim_out[1]);

	printf("\n");

	printf("kernel_size: %d\n", conv_layer->kernel_size);
	printf("strides: %d\n", conv_layer->strides);

	printf("\n");

	print_conv_kernels(conv_layer);
	print_conv_bias(conv_layer);

}

void load_dense_relu_test(){

	printf("---------- load dense relu test ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_dense_relu_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	dense_struct* dense_layer = malloc(sizeof(dense_struct)); // reserves memory
    load_dense(dense_layer, file_ptr, file_idx); // load layer

	printf("dim_in: %d\n", dense_layer->dim_in);
	printf("dim_out: %d\n", dense_layer->dim_out);
	printf("activation function: %s\n", dense_layer->actFunc);

	printf("\n");

	print_dense_kernel(dense_layer);
	print_dense_bias(dense_layer);

}

void load_dense_softmax_test(){
	
	printf("---------- load dense softmax test ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_dense_softmax_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	dense_struct* dense_layer = malloc(sizeof(dense_struct)); // reserves memory
    load_dense(dense_layer, file_ptr, file_idx); // load layer

	printf("dim_in: %d\n", dense_layer->dim_in);
	printf("dim_out: %d\n", dense_layer->dim_out);
	printf("activation function: %s\n", dense_layer->actFunc);

	printf("\n");

	print_dense_kernel(dense_layer);
	print_dense_bias(dense_layer);

}

void load_flatten_test(){

	printf("---------- load flatten test ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_flatten_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	flatten_struct* flatten_layer = malloc(sizeof(flatten_struct));
    load_flatten(flatten_layer, file_ptr, file_idx); 
	
	printf("= Load Flatten Layer==============\n"); 
	printf("dim_in: %u x %u x %u \n",  flatten_layer->dim_in[0], flatten_layer->dim_in[1], flatten_layer->dim_in[2]);
	printf("dim_out: %u \n",  flatten_layer->dim_out);  
}

void load_maxpool_test(){

	printf("---------- load maxpool test ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_maxpool_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	maxpool_struct* maxpool_layer = malloc(sizeof(maxpool_struct)); 
	load_maxpool(maxpool_layer, file_ptr,  file_idx); 

	printf("= Load Maxpool Layer==============\n"); 
	printf("dim_in: %u x %u \n",  maxpool_layer->dim_in[0], maxpool_layer->dim_in[1]);
	printf("dim_out: %u x %u \n",  maxpool_layer->dim_out[0], maxpool_layer->dim_out[1]);
	printf("pool_size: %u \n", maxpool_layer->pool_size);
	printf("strides: %u \n", maxpool_layer->strides);  
}

// Complete layers tests

void conv_test_completo(){

	printf("---------- conv test completo ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_conv_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	conv_struct* conv_layer = malloc(sizeof(conv_struct)); // reserves memory
    load_conv(conv_layer, file_ptr, file_idx); // load layer

	// Load .txt image file
	FILE* image_ptr = fopen("../modelos_redes/test/conv_test_img.txt", "r");

	if(image_ptr == NULL){
        printf("Error al cargar la imagen\n\n");  
        exit(-1); 
    }
	else{
		printf("image ptr ok!\n");
	}

	// Init image_idx
    long int* image_idx = malloc(sizeof(long int));
    *image_idx = 0;

	generic_images images = malloc(conv_layer->dim_in[0] * sizeof(generic_image));
	load_image(images, image_ptr, image_idx);

	// Convolution function
	generic_images res = conv(images, conv_layer);

	print_images(res, conv_layer->dim_out); // print result

}

void maxpool_test_completo(){

	printf("---------- maxpool test completo ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_maxpool_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	maxpool_struct* maxpool_layer = malloc(sizeof(maxpool_struct)); 
	load_maxpool(maxpool_layer, file_ptr,  file_idx); 
	
	FILE* img_ptr = fopen("../modelos_redes/test/maxpool_test_img.txt", "r");

	if(img_ptr == NULL){
        printf("Error al cargar la imagen\n\n");  
        exit(-1); 
    }
	else{
		printf("image ptr ok!\n");
	}

	long int* img_idx = malloc(sizeof(long int));
    *img_idx = 0;

	generic_images imgs = malloc(sizeof(generic_image)* maxpool_layer->dim_in[0]);
	load_image(imgs, img_ptr, img_idx); 

	generic_images res = maxpool(imgs, maxpool_layer);
	print_images(res, maxpool_layer->dim_out);  
}

void flatten_test_completo(){

	printf("---------- flatten test completo ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_flatten_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	flatten_struct* flatten_layer = malloc(sizeof(flatten_struct)); 
	load_flatten(flatten_layer, file_ptr,  file_idx); 
	
	FILE* img_ptr = fopen("../modelos_redes/test/flatten_test_img.txt", "r"); 
	if(img_ptr == NULL){
		printf("Cannot open file"); 
		exit(-1); 
	}
	long int* img_idx = malloc(sizeof(long int));
    *img_idx = 0;

	generic_images imgs = malloc(sizeof(generic_image)* flatten_layer->dim_in[0]);
	load_image(imgs, img_ptr, img_idx); 

	float* res = flatten(imgs,  flatten_layer); 
	print_vect(res, flatten_layer->dim_out); 
	
}

void dense_relu_test_completo(){

	printf("---------- dense relu test completo ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_dense_relu_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}

	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	dense_struct* dense_layer = malloc(sizeof(dense_struct)); 
	load_dense(dense_layer, file_ptr,  file_idx); 
	
	FILE* img_ptr = fopen("../modelos_redes/test/dense_relu_test_img.txt", "r"); 
	if(img_ptr == NULL){
		printf("Cannot open file"); 
		exit(-1); 
	}
	long int* img_idx = malloc(sizeof(long int));
    *img_idx = 0;

	float *in_vect = malloc(sizeof(float)*dense_layer->dim_in); 
	
	load_vect(in_vect, img_ptr, img_idx);

	float* res = dense(in_vect,  dense_layer); 
	print_vect(res, dense_layer->dim_out); 
	
}

void dense_softmax_test_completo(){

	printf("---------- dense softmax test completo ----------\n");

	FILE* file_ptr = fopen("../modelos_redes/test/load_dense_softmax_test.orga", "r");

	if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}
	
	// Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;
    
    // Read n_layers
    int n_layers = read_int(file_ptr, file_idx);

	*file_idx += 2; // next line

	*file_idx += 2; // dont read title

	dense_struct* dense_layer = malloc(sizeof(dense_struct)); 
	load_dense(dense_layer, file_ptr,  file_idx); 
	
	FILE* img_ptr = fopen("../modelos_redes/test/dense_softmax_test_img.txt", "r"); 
	if(img_ptr == NULL){
		printf("Cannot open file"); 
		exit(-1); 
	}
	long int* img_idx = malloc(sizeof(long int));
    *img_idx = 0;

	float *in_vect = malloc(sizeof(float)*dense_layer->dim_in); 
	
	load_vect(in_vect, img_ptr, img_idx);  

	float* res = dense(in_vect,  dense_layer); 
	print_vect(res, dense_layer->dim_out); 
	
}

// load complete test (load net)

void load_net_test_completo(){

	printf("---------- load net test completo ----------\n");

	// Read file
    FILE* file_ptr = fopen("../modelos_redes/test/weight_to_file.orga", "r"); 
    
    // Check file_ptr
    if(file_ptr == NULL){
        printf("Error al cargar la red\n\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}
    printf("red file cargada \n");
    
    // Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;

	net* loaded_net = malloc(sizeof(net));
	loadNet(loaded_net, file_ptr, file_idx);

	for(int i = 0; i < loaded_net->cant_capas; i++){

		printf("%c ",loaded_net->arq_names[i]);
	}

	printf("\n");

	conv_struct* c_1 = loaded_net->arq_structs[0];
	printf("dim in: %d %d\n", c_1->dim_in[0], c_1->dim_in[1]);
	printf("dim out: %d %d\n", c_1->dim_out[0], c_1->dim_out[1]);

	maxpool_struct* m_1 = loaded_net->arq_structs[1];
	printf("dim in: %d %d\n", m_1->dim_in[0], m_1->dim_in[1]);
	printf("dim out: %d %d\n", m_1->dim_out[0], m_1->dim_out[1]);

	conv_struct* c_2 = loaded_net->arq_structs[2];
	printf("dim in: %d %d\n", c_2->dim_in[0], c_2->dim_in[1]);
	printf("dim out: %d %d\n", c_2->dim_out[0], c_2->dim_out[1]);

	maxpool_struct* m_2 = loaded_net->arq_structs[3];
	printf("dim in: %d %d\n", m_2->dim_in[0], m_2->dim_in[1]);
	printf("dim out: %d %d\n", m_2->dim_out[0], m_2->dim_out[1]);

	flatten_struct* f_1 = loaded_net->arq_structs[4];
	printf("dim in: %d %d\n", f_1->dim_in[0], f_1->dim_in[1]);
	printf("dim out: %d\n", f_1->dim_out);

	dense_struct* d_1 = loaded_net->arq_structs[5];
	printf("dim in: %d\n", d_1->dim_in);
	printf("dim out: %d\n", d_1->dim_out);

	dense_struct* d_2 = loaded_net->arq_structs[6];
	printf("dim in: %d\n", d_2->dim_in);
	printf("dim out: %d\n", d_2->dim_out);
}

void execute_net_test_completo(){

	printf("---------- execute net test completo ----------\n");

	// Read file
    FILE* file_ptr = fopen("../modelos_redes/test/kaggel_model.orga", "r"); 
    
    // Check file_ptr
    if(file_ptr == NULL){
        printf("Error al cargar la red\n");  
        exit(-1); 
    }
	else{
		printf("file ptr ok!\n");
	}
	
    printf("red file cargada \n");
    
    // Init file_idx
    long int* file_idx = malloc(sizeof(long int));
    *file_idx = 0;

	net* loaded_net = malloc(sizeof(net)); 
	loadNet(loaded_net, file_ptr, file_idx);

	// Load .img image file
	FILE* image_ptr = fopen("../modelos_redes/test/execute_net_test_img.txt", "r");

	// Init file_idx
    long int* image_idx = malloc(sizeof(long int));
    *image_idx = 0;

	generic_images images = malloc(sizeof(generic_image));
	load_image(images, image_ptr, image_idx);

	float* res = executeNet(loaded_net, images);
	print_vect(res,10);

}

#endif