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

using namespace std;

/* Serves as the initial round during encryption
 * AddRoundKey is simply an XOR of a 128-bit block with the 128-bit key.
 */
__device__ void cudaAddRoundKey(unsigned char* state, unsigned char* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* Perform substitution to each of the 16 bytes
 * Uses S-box as lookup table
 */
__device__ void cudaSubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = cudaS[state[i]];
        //printf("Hello, World! from thread %d with state[i] %d\n", i, state[i]);
    }
    
}


// Shift left, adds diffusion
__device__ void cudaShiftRows(unsigned char* state) {
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
__device__ void cudaMixColumns(unsigned char* state) {
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
__device__ void cudaRound(unsigned char* state, unsigned char* key) {
    //cudaSubBytes << <1, 16 >> > (state);
    cudaSubBytes(state);
	cudaShiftRows(state);
	cudaMixColumns(state);
	cudaAddRoundKey(state, key);
}

// Same as Round() except it doesn't mix columns
__device__ void cudaFinalRound(unsigned char* state, unsigned char* key) {
    //cudaSubBytes << <1, 16 >> > (state);
    cudaSubBytes(state);
	cudaShiftRows(state);
	cudaAddRoundKey(state, key);
}

/* The AES encryption function
 * Organizes the confusion and diffusion steps into one function
 */
__global__ void cudaAESEncrypt(unsigned char* paddedMessage, unsigned char* expandedKey, unsigned char* encryptedMessage, int paddedMessageLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numBlocks = paddedMessageLen / 16;

    if (idx < numBlocks) {
        unsigned char state[16];
        for (int i = 0; i < 16; i++) {
            state[i] = paddedMessage[idx * 16 + i];
        }

        cudaAddRoundKey(state, expandedKey); // Initial round

        for (int i = 0; i < 9; i++) {
            cudaRound(state, expandedKey + (16 * (i + 1)));
        }

        cudaFinalRound(state, expandedKey + 160);

        // Copy encrypted state to buffer
        for (int i = 0; i < 16; i++) {
            encryptedMessage[idx * 16 + i] = state[i];
        }
    }
}

string cudaEncrypt(string inputFileName) {
    std::cout << "=======================================" << std::endl;
    std::cout << " 128-bit AES CUDA 11.6 Encryption Tool   " << std::endl;
    std::cout << "=======================================" << std::endl;

    double start, end, total=0;
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

    // CUDA parameters
    int blockSize = 256; // Adjust block size as needed

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

        int numBlocks = (paddedMessageLen + blockSize - 1) / blockSize;
        start = omp_get_wtime();
        // Launch CUDA kernel
        cudaAESEncrypt << <numBlocks, blockSize >> > (d_paddedMessage, d_expandedKey, d_encryptedMessage, paddedMessageLen);
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
