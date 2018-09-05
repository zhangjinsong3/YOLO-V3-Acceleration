#include <interpPlugin.h>


dim3 cuda_gridsize(unsigned int n){
	unsigned int k = (n-1) / BLOCK + 1;
	unsigned int x = k;
	unsigned int y = 1;
	if(x > 65535){
		x = ceil(sqrt(k));
		y = (n-1)/(x*BLOCK) + 1;
	}
	dim3 d = {x, y, 1};
	return d;
} 

/* nearest neighbor upsampling used in darknet*/
__global__ void upsample_gpu(int N, const float *x, int w, int h, int c, int batch, int zoomFactor, float *out, const char* mode="nearest")
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(i >= N) return;
	int out_index = i;
	int out_w = i%(w*zoomFactor);
	i = i/(w*zoomFactor);
	int out_h = i%(h*zoomFactor);
	i = i/(h*zoomFactor);
	int _c = i%c;
	i = i/_c;
	int _b = i%batch;
	int in_w = out_w/zoomFactor;
	int in_h = out_h/zoomFactor;
	int in_offset = _b*c*w*h + _c*w*h;
	int in_index00 = in_offset + in_h*w + in_w;
	if(mode == "bilinear"){
		int in_index01 = (in_w+1 > w) ? in_index00 : (in_index00 + 1);
		int in_index10 = (in_h+1 > h) ? in_index00 : (in_index00 + w);
		int in_index11 = (in_index01 == in_index10) ? in_index00 : (in_index10 + 1);
		
		float u = (float)(out_h % zoomFactor)/zoomFactor;
		float v = (float)(out_w % zoomFactor)/zoomFactor;
		out[out_index] = (1-u)*(1-v)*x[in_index00] + \
										 (1-u)*v*x[in_index01] + \
										 u*(1-v)*x[in_index10] + \
										 u*v*x[in_index11];
	}
	else if(mode == "nearest"){
		out[out_index] = x[in_index00];
	}
}

void interp_gpu(const float *x, int w, int h, int c, int batch, int zoomFactor, float *out, cudaStream_t stream)
{
	int outSize = w*zoomFactor*h*zoomFactor*c*batch;
	upsample_gpu<<<cuda_gridsize(outSize), BLOCK, 0, stream>>>(outSize, x, w, h, c, batch, zoomFactor, out);
}
