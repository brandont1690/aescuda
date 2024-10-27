#pragma once
#include <omp.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.cuh"
#include <crt/device_functions.h>
#include <crt/device_functions.h>

using namespace std;

/* Serves as the initial round during encryption
 * AddRoundKey is simply an XOR of a 128-bit block with the 128-bit key.
 */
__device__ void testCudaAddRoundKey(unsigned char* state, unsigned char* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* Perform substitution to each of the 16 bytes
 * Uses S-box as lookup table
 */
__device__ void testCudaSubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = cudaS[state[i]];
        //printf("Hello, World! from thread %d with state[i] %d\n", i, state[i]);
    }

}


// Shift left, adds diffusion
__device__ void testCudaShiftRows(unsigned char* state) {
    unsigned char tmp[16];

    tmp[0] = state[0];
    tmp[1] = state[5];
    tmp[2] = state[10];
    tmp[3] = state[15];

    tmp[4] = state[4];
    tmp[5] = state[9];
    tmp[6] = state[14];
    tmp[7] = state[3];

    tmp[8] = state[8];
    tmp[9] = state[13];
    tmp[10] = state[2];
    tmp[11] = state[7];

    tmp[12] = state[12];
    tmp[13] = state[1];
    tmp[14] = state[6];
    tmp[15] = state[11];

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

/* MixColumns uses mul2, mul3 look-up tables
 * Source of diffusion
 */
#define N 4  // Dimension of the matrices

__device__ unsigned char gfMul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    unsigned char hiBitSet;
    for (int counter = 0; counter < 8; counter++) {
        if (b & 1) {
            p ^= a;
        }
        hiBitSet = (a & 0x80);
        a <<= 1;
        if (hiBitSet) {
            a ^= 0x1B; // Polynomial used in AES
        }
        b >>= 1;
    }
    return p;
}

__device__ void testCudaMixColumns(unsigned char* state) {
    unsigned char tmp[16];

    tmp[0] = (unsigned char)cudaMul2[state[0]] ^ cudaMul3[state[1]] ^ state[2] ^ state[3];
    tmp[1] = (unsigned char)state[0] ^ cudaMul2[state[1]] ^ cudaMul3[state[2]] ^ state[3];
    tmp[2] = (unsigned char)state[0] ^ state[1] ^ cudaMul2[state[2]] ^ cudaMul3[state[3]];
    tmp[3] = (unsigned char)cudaMul3[state[0]] ^ state[1] ^ state[2] ^ cudaMul2[state[3]];

    tmp[4] = (unsigned char)cudaMul2[state[4]] ^ cudaMul3[state[5]] ^ state[6] ^ state[7];
    tmp[5] = (unsigned char)state[4] ^ cudaMul2[state[5]] ^ cudaMul3[state[6]] ^ state[7];
    tmp[6] = (unsigned char)state[4] ^ state[5] ^ cudaMul2[state[6]] ^ cudaMul3[state[7]];
    tmp[7] = (unsigned char)cudaMul3[state[4]] ^ state[5] ^ state[6] ^ cudaMul2[state[7]];

    tmp[8] = (unsigned char)cudaMul2[state[8]] ^ cudaMul3[state[9]] ^ state[10] ^ state[11];
    tmp[9] = (unsigned char)state[8] ^ cudaMul2[state[9]] ^ cudaMul3[state[10]] ^ state[11];
    tmp[10] = (unsigned char)state[8] ^ state[9] ^ cudaMul2[state[10]] ^ cudaMul3[state[11]];
    tmp[11] = (unsigned char)cudaMul3[state[8]] ^ state[9] ^ state[10] ^ cudaMul2[state[11]];

    tmp[12] = (unsigned char)cudaMul2[state[12]] ^ cudaMul3[state[13]] ^ state[14] ^ state[15];
    tmp[13] = (unsigned char)state[12] ^ cudaMul2[state[13]] ^ cudaMul3[state[14]] ^ state[15];
    tmp[14] = (unsigned char)state[12] ^ state[13] ^ cudaMul2[state[14]] ^ cudaMul3[state[15]];
    tmp[15] = (unsigned char)cudaMul3[state[12]] ^ state[13] ^ state[14] ^ cudaMul2[state[15]];

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

/* Each round operates on 128 bits at a time
 * The number of rounds is defined in AESEncrypt()
 */
__device__ void testCudaRound(unsigned char* state, unsigned char* key) {
    testCudaSubBytes(state);
    testCudaShiftRows(state);

    dim3 threadsPerBlock(N, N);  // Each thread handles one element of the state matrix
    //testCudaMixColumns << <1, threadsPerBlock >> > (state);
    cudaDeviceSynchronize();

    testCudaAddRoundKey(state, key);
}

// Same as Round() except it doesn't mix columns
__device__ void testCudaFinalRound(unsigned char* state, unsigned char* key) {
    //cudaSubBytes << <1, 16 >> > (state);
    testCudaSubBytes(state);
    testCudaShiftRows(state);
    testCudaAddRoundKey(state, key);
}

/* The AES encryption function
 * Organizes the confusion and diffusion steps into one function
 */
__global__ void testCudaAESEncrypt(unsigned char* paddedMessage, unsigned char* expandedKey, unsigned char* encryptedMessage, int paddedMessageLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numBlocks = paddedMessageLen / 16;

    if (idx < numBlocks) {
        unsigned char state[16];
        for (int i = 0; i < 16; i++) {
            state[i] = paddedMessage[idx * 16 + i];
        }

        testCudaAddRoundKey(state, expandedKey); // Initial round

        for (int round = 0; round < 1; round++) {
            // Debug prints for the starting state
            //if (threadIdx.x == 0 && threadIdx.y == 0) {
            //    for (int i = 0; i < 16; i++) {
            //        printf("The starting state[%d]: %u\n", i, state[i]);
            //    }
            //}
            //__syncthreads();

            // Separate local state array for each thread
            unsigned char localState[16];
            for (int i = 0; i < 16; i++) {
                localState[i] = state[i];
            }

            testCudaSubBytes(localState);
            testCudaShiftRows(localState);

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int i = 0; i < 16; i++) {
                    printf("The starting state[%d]: %u\n", i, localState[i]);
                }
            }

            __syncthreads();

            // Ensure only the 4 threads in Y dimension are used
            if (threadIdx.y < 4) {
                printf("Thread %d executing MixColumns for round Y %d\n", threadIdx.y, round);
                int element = threadIdx.y * 4;

                if (threadIdx.y == 0) {
                    localState[0] =     (unsigned char)cudaMul2[localState[0]] ^ cudaMul3[localState[1]] ^ localState[2] ^ localState[3];
                    localState[1] = (unsigned char)localState[0] ^ cudaMul2[localState[1]] ^ cudaMul3[localState[2]] ^ localState[3];
                    localState[2] = (unsigned char)localState[0] ^ localState[1] ^ cudaMul2[localState[2]] ^ cudaMul3[localState[3]];
                    localState[3] = (unsigned char)cudaMul3[localState[0]] ^ localState[1] ^ localState[2] ^ cudaMul2[localState[3]];
                    for (int i = 0; i < 4; i++) {
                        printf("The state[%d]: %u\n", i, localState[i]);
                    }
                }
                else if (threadIdx.y == 1) {
                    localState[4] = (unsigned char)cudaMul2[localState[4]] ^ cudaMul3[localState[5]] ^ localState[6] ^ localState[7];
                    localState[5] = (unsigned char)localState[4] ^ cudaMul2[localState[5]] ^ cudaMul3[localState[6]] ^ localState[7];
                    localState[6] = (unsigned char)localState[4] ^ localState[5] ^ cudaMul2[localState[6]] ^ cudaMul3[localState[7]];
                    localState[7] = (unsigned char)cudaMul3[localState[4]] ^ localState[5] ^ localState[6] ^ cudaMul2[localState[7]];
                    for (int i = 4; i < 8; i++) {
                        printf("The state[%d]: %u\n", i, localState[i]);
                    }
                }
                else if (threadIdx.y == 2) {
                    localState[8] = (unsigned char)cudaMul2[localState[8]] ^ cudaMul3[localState[9]] ^ localState[10] ^ localState[11];
                    localState[9] = (unsigned char)localState[8] ^ cudaMul2[localState[9]] ^ cudaMul3[localState[10]] ^ localState[11];
                    localState[10] = (unsigned char)localState[8] ^ localState[9] ^ cudaMul2[localState[10]] ^ cudaMul3[localState[11]];
                    localState[11] = (unsigned char)cudaMul3[localState[8]] ^ localState[9] ^ localState[10] ^ cudaMul2[localState[11]];
                    for (int i = 8; i < 12; i++) {
                        printf("The state[%d]: %u\n", i, localState[i]);
                    }
                }
                else if (threadIdx.y == 3) {
                    localState[12] = (unsigned char)cudaMul2[localState[12]] ^ cudaMul3[localState[13]] ^ localState[14] ^ localState[15];
                    localState[13] = (unsigned char)localState[12] ^ cudaMul2[localState[13]] ^ cudaMul3[localState[14]] ^ localState[15];
                    localState[14] = (unsigned char)localState[12] ^ localState[13] ^ cudaMul2[localState[14]] ^ cudaMul3[localState[15]];
                    localState[15] = (unsigned char)cudaMul3[localState[12]] ^ localState[13] ^ localState[14] ^ cudaMul2[localState[15]];
                    for (int i = 12; i < 16; i++) {
                        printf("The state[%d]: %u\n", i, localState[i]);
                    }
                }
            }

            __syncthreads(); // Ensure all threads have finished MixColumns

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int i = 0; i < 16; i++) {
                    printf("The ending state[%d]: %u\n", i, localState[i]);
                }
            }
            __syncthreads();

            // Copy the local state back to the main state
            for (int i = 0; i < 16; i++) {
                state[i] = localState[i];
            }

            // Debug prints for the ending state
            //if (threadIdx.x == 0 && threadIdx.y == 0) {
            //    for (int i = 0; i < 16; i++) {
            //        printf("The ending state[%d]: %u\n", i, state[i]);
            //    }
            //}
            testCudaAddRoundKey(state, expandedKey + (16 * (round + 1)));
        }

        testCudaSubBytes(state);
        testCudaShiftRows(state);
        testCudaAddRoundKey(state, expandedKey + 160);

        // Copy encrypted state to buffer
        for (int i = 0; i < 16; i++) {
            encryptedMessage[idx * 16 + i] = state[i];
        }
    }
}

string testCudaEncrypt(string inputFileName) {
    std::cout << "=======================================" << std::endl;
    std::cout << " 128-bit AES CUDA 11.6 Encryption Tool   " << std::endl;
    std::cout << "=======================================" << std::endl;

    double start, end, total = 0;
    double kstart, kend;
    double fullstart = omp_get_wtime();

    // Define chunk size (adjust as needed)
    const size_t chunkSize = 1024 * 1024 * 100; // 500 MB

    // Read in the message from input file
    std::ifstream infile(inputFileName, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Unable to open input file: " << inputFileName << std::endl;
        return "";
    }

    // Load key from file
    std::string keyStr;
    std::ifstream keyfile("keyfile", std::ios::binary);
    if (!keyfile.is_open()) {
        std::cout << "Unable to open file keyfile" << std::endl;
        infile.close();
        return "";
    }

    getline(keyfile, keyStr);
    keyfile.close();

    // Convert the key to unsigned char array
    unsigned char key[16];
    std::istringstream hexCharsStream(keyStr);
    for (int i = 0; i < 16; ++i) {
        int c;
        hexCharsStream >> std::hex >> c;
        key[i] = static_cast<unsigned char>(c);
    }

    // Expand the key
    unsigned char expandedKey[176];
    kstart = omp_get_wtime();
    cudaKeyExpansion(key, expandedKey);
    kend = omp_get_wtime() - kstart;
    std::cout << "Time taken for key expansion: " << kend << std::endl;

    // Allocate device memory
    unsigned char* d_expandedKey;
    cudaMalloc((void**)&d_expandedKey, 176 * sizeof(unsigned char)); // 176 is the size of expanded key
    cudaMemcpy(d_expandedKey, expandedKey, 176 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Process the input file in chunks
    std::string outputFileName = inputFileName.substr(0, inputFileName.find_last_of(".")) + ".aes";
    std::ofstream outfile(outputFileName, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Unable to open file for writing encrypted message" << std::endl;
        infile.close();
        cudaFree(d_expandedKey);
        return "";
    }

    // Loop through the input file and encrypt each chunk
    while (!infile.eof()) {
        // Allocate memory for chunk buffer
        unsigned char* chunk = new unsigned char[chunkSize];
        infile.read(reinterpret_cast<char*>(chunk), chunkSize);
        size_t bytesRead = infile.gcount();

        // Pad message to 16 bytes
        int originalLen = bytesRead;
        int paddedMessageLen = originalLen;
        int paddingLength = 0;
        if ((paddedMessageLen % 16) != 0) {
            paddingLength = 16 - (paddedMessageLen % 16);
            paddedMessageLen += paddingLength;
        }

        unsigned char* paddedMessage = new unsigned char[paddedMessageLen];
        for (int i = 0; i < paddedMessageLen; i++) {
            if (i >= originalLen) {
                paddedMessage[i] = paddingLength;
            }
            else {
                paddedMessage[i] = chunk[i];
            }
        }

        unsigned char* encryptedMessage = new unsigned char[paddedMessageLen];
        unsigned char* d_paddedMessage;
        unsigned char* d_encryptedMessage;
        cudaMalloc((void**)&d_paddedMessage, paddedMessageLen * sizeof(unsigned char));
        cudaMalloc((void**)&d_encryptedMessage, paddedMessageLen * sizeof(unsigned char));
        cudaMemcpy(d_paddedMessage, paddedMessage, paddedMessageLen * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // CUDA parameters
        dim3 threadsPerBlock(256, 4); // Adjust block size as needed
        dim3  numBlocks ((paddedMessageLen + 256 - 1) / 256);

        start = omp_get_wtime();
        // Launch CUDA kernel
        testCudaAESEncrypt << <numBlocks, threadsPerBlock >> > (d_paddedMessage, d_expandedKey, d_encryptedMessage, paddedMessageLen);
        cudaDeviceSynchronize();
        end = omp_get_wtime() - start;
        // Copy encrypted message back to host
        cudaMemcpy(encryptedMessage, d_encryptedMessage, paddedMessageLen * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Write encrypted chunk to output file
        outfile.write(reinterpret_cast<char*>(encryptedMessage), paddedMessageLen);

        // Free memory
        delete[] chunk;
        delete[] paddedMessage;
        delete[] encryptedMessage;
        cudaFree(d_paddedMessage);
        cudaFree(d_encryptedMessage);
        total += end;
    }


    // Close files
    infile.close();
    outfile.close();
    double fullend = omp_get_wtime() - fullstart;
    std::cout << "Wrote encrypted message to file: " << outputFileName << std::endl;
    std::cout << "Time taken to perform AES encryption: " << total << std::endl;
    std::cout << "Total Time taken for the entire process: " << fullend << std::endl;

    // Free memory
    cudaFree(d_expandedKey);
    return outputFileName;
}
