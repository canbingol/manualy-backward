#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// XOR-gate
    float train[][3] = {
        {0,0,0},
        {1,0,1},
        {0,1,1},
        {1,1,0},
    };

    float sigmoidf(float w){
        return (1 / (1+ exp(-w)));
    }

    float rand_float(){
        return (float) rand() / (float) RAND_MAX; 
    }

#define train_count sizeof(train) / sizeof(train[0])

    float cost(float w11, float w12, float b1,float w21, float b2){

    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float z1 = w11*x1 + w12*x2 +b1;
        float a1 = sigmoidf(z1);

        float z2 = w21*a1+b2;
        float y = sigmoidf(z2);

        float d = y - train[i][2];
        result += d*d;
    }
    
    result /= train_count;
    return result;

    };

    int main(void)
    {

        float eps = 1e-3;
        float rate = 1e-3;
        float w11 = rand_float();
        float w12 = rand_float();
        float b1 = rand_float();

        float w21 = rand_float();
        float w22 = rand_float();
        float b2 = rand_float();

        for (size_t i = 0; i < 5000; i++)
        {

            float c = cost(w11,w12,b1,w21,b2);

            float dw11 = (cost(w11+eps, w12,b1,w21,b2) - c) / eps;
            float dw12 = (cost(w11, w12+ eps,b1,w21,b2) - c) / eps;
            float db1 = (cost(w11,w12,b1+eps,w21,b2) - c) / eps;
            float dw21 = (cost(w11,w12,b1,w21+eps,b2) - c) / eps;
            float db2 = (cost(w11,w12,b1,w21,b2+eps) - c) / eps;

            w11 -= rate*dw11;
            w12 -= rate *dw12;
            b1 -= rate*db1;
            w21 -= rate *dw21;
            b2 -= rate*db2;
        }
        printf("w11 = %f, w12 = %f, b1 = %f,w21 = %f, b2 = %f ,cost = %f\n ", w11,w12,b1,w21,b2,cost(w11,w12,b1,w21,b2));
        return 0;
    }