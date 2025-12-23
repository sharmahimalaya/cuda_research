#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>

using namespace std;

#define NUM_ELEMENTS 4
#define NUM_THREADS 256
#define NUM_BLOCKS 32

__global__ void add(float4 *a, float4 *b, float4* c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<NUM_ELEMENTS){
        float4 a1 = a[idx];
        float4 a2 = b[idx];
        printf("a1: %f, %f, %f, %f\n", a1.x, a1.y,a1.z, a1.w);
        printf("a2: %f, %f, %f, %f\n", a2.x, a2.y,a2.z, a2.w);
        float4 r;
        r.x = a1.x + a2.x;
        r.y = a1.y + a2.y;
        r.z = a1.z + a2.z;
        r.w = a1.w + a2.w;
        printf("r: %f, %f, %f, %f\n", r.x, r.y,r.z, r.w);
        c[idx] = r;

    }
}

int main(){
    float4 h_a[NUM_ELEMENTS], h_b[NUM_ELEMENTS], h_c[NUM_ELEMENTS];
    //initialising data
    for(int i=0;i<NUM_ELEMENTS;i++){
        h_a[i] = make_float4(i,i,i,i);
        h_b[i] = make_float4(0,0,0,0);
    }
    for(int i=0;i<NUM_ELEMENTS;i++){
        cout << "A : [" << i << "] = (" << h_a[i].x << ", " << h_a[i].y << ", " << h_a[i].z << ", " << h_a[i].w << ")" << endl;
    }
    for(int i=0;i<NUM_ELEMENTS;i++){
        cout << "B : [" << i << "] = (" << h_b[i].x << ", " << h_b[i].y << ", " << h_b[i].z << ", " << h_b[i].w << ")" << endl;
    }
    float4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, NUM_ELEMENTS*sizeof(float4));
    cudaMalloc(&d_b, NUM_ELEMENTS*sizeof(float4));
    cudaMalloc(&d_c, NUM_ELEMENTS*sizeof(float4));

    cudaMemcpy(d_a,h_a,NUM_ELEMENTS*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,NUM_ELEMENTS*sizeof(float4), cudaMemcpyHostToDevice);

    add<<<NUM_BLOCKS,NUM_THREADS>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,NUM_ELEMENTS*sizeof(float4), cudaMemcpyDeviceToHost);
    for(int i=0;i<NUM_ELEMENTS;i++){
        cout << "Result : [" << i << "] = (" << h_c[i].x << ", " << h_c[i].y << ", " << h_c[i].z << ", " << h_c[i].w << ")" << endl;
    }
    return 0;
}