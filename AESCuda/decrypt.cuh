#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <omp.h> // Include OpenMP for omp_get_wtime()
#include "structures.cuh"

using namespace std;

/* Used in Round() and serves as the final round during decryption
 * SubRoundKey is simply an XOR of a 128-bit block with the 128-bit key.
 * So basically does the same as AddRoundKey in the encryption
 */
__device__ void cudaSubRoundKey(unsigned char* state, unsigned char* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* InverseMixColumns uses mul9, mul11, mul13, mul14 look-up tables
 * Unmixes the columns by reversing the effect of MixColumns in encryption
 */
__device__ void cudaInverseMixColumns(unsigned char* state) {
    unsigned char tmp[16];

    tmp[0] = (unsigned char)cudaMul14[state[0]] ^   cudaMul11[state[1]] ^ cudaMul13[state[2]] ^ cudaMul9[state[3]];
    tmp[1] = (unsigned char)cudaMul9[state[0]] ^    cudaMul14[state[1]] ^ cudaMul11[state[2]] ^ cudaMul13[state[3]];
    tmp[2] = (unsigned char)cudaMul13[state[0]] ^   cudaMul9[state[1]] ^  cudaMul14[state[2]] ^ cudaMul11[state[3]];
    tmp[3] = (unsigned char)cudaMul11[state[0]] ^   cudaMul13[state[1]] ^ cudaMul9[state[2]] ^  cudaMul14[state[3]];

    tmp[4] = (unsigned char)cudaMul14[state[4]] ^ cudaMul11[state[5]] ^ cudaMul13[state[6]] ^ cudaMul9[state[7]];
    tmp[5] = (unsigned char)cudaMul9[state[4]] ^  cudaMul14[state[5]] ^ cudaMul11[state[6]] ^ cudaMul13[state[7]];
    tmp[6] = (unsigned char)cudaMul13[state[4]] ^ cudaMul9[state[5]] ^  cudaMul14[state[6]] ^ cudaMul11[state[7]];
    tmp[7] = (unsigned char)cudaMul11[state[4]] ^ cudaMul13[state[5]] ^ cudaMul9[state[6]] ^  cudaMul14[state[7]];

    tmp[8] = (unsigned char) cudaMul14[state[8]] ^ cudaMul11[state[9]] ^ cudaMul13[state[10]] ^ cudaMul9[state[11]];
    tmp[9] = (unsigned char) cudaMul9[state[8]] ^  cudaMul14[state[9]] ^ cudaMul11[state[10]] ^ cudaMul13[state[11]];
    tmp[10] = (unsigned char)cudaMul13[state[8]] ^ cudaMul9[state[9]] ^  cudaMul14[state[10]] ^ cudaMul11[state[11]];
    tmp[11] = (unsigned char)cudaMul11[state[8]] ^ cudaMul13[state[9]] ^ cudaMul9[state[10]] ^  cudaMul14[state[11]];

    tmp[12] = (unsigned char)cudaMul14[state[12]] ^ cudaMul11[state[13]] ^ cudaMul13[state[14]] ^ cudaMul9[state[15]];
    tmp[13] = (unsigned char)cudaMul9[state[12]] ^  cudaMul14[state[13]] ^ cudaMul11[state[14]] ^ cudaMul13[state[15]];
    tmp[14] = (unsigned char)cudaMul13[state[12]] ^ cudaMul9[state[13]] ^  cudaMul14[state[14]] ^ cudaMul11[state[15]];
    tmp[15] = (unsigned char)cudaMul11[state[12]] ^ cudaMul13[state[13]] ^ cudaMul9[state[14]] ^  cudaMul14[state[15]];

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

// Shifts rows right (rather than left) for decryption
__device__ void cudaDecShiftRows(unsigned char* state) {
    unsigned char tmp[16];

    /* Column 1 */
    tmp[0] = state[0];
    tmp[1] = state[13];
    tmp[2] = state[10];
    tmp[3] = state[7];

    /* Column 2 */
    tmp[4] = state[4];
    tmp[5] = state[1];
    tmp[6] = state[14];
    tmp[7] = state[11];

    /* Column 3 */
    tmp[8] = state[8];
    tmp[9] = state[5];
    tmp[10] = state[2];
    tmp[11] = state[15];

    /* Column 4 */
    tmp[12] = state[12];
    tmp[13] = state[9];
    tmp[14] = state[6];
    tmp[15] = state[3];

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

/* Perform substitution to each of the 16 bytes
 * Uses inverse S-box as lookup table
 */
__device__ void cudaDecSubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) { // Perform substitution to each of the 16 bytes
        state[i] = cudaInv_s[state[i]];
    }
}

/* Each round operates on 128 bits at a time
 * The number of rounds is defined in AESDecrypt()
 * Not surprisingly, the steps are the encryption steps but reversed
 */
__device__ void cudaDecRound(unsigned char* state, unsigned char* key) {
    cudaSubRoundKey(state, key);
    cudaInverseMixColumns(state);
    cudaDecShiftRows(state);
    cudaDecSubBytes(state);
}

// Same as Round() but no InverseMixColumns
__device__ void cudaInitialRound(unsigned char* state, unsigned char* key) {
    cudaSubRoundKey(state, key);
    cudaDecShiftRows(state);
    cudaDecSubBytes(state);
}

/* The AES decryption function
 * Organizes all the decryption steps into one function
 */
__global__ void cudaAESDecrypt(const unsigned char* encryptedMessage, unsigned char* expandedKey, unsigned char* decryptedMessage, int messageLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = idx * 16;

    if (startIdx < messageLen) {
        unsigned char state[16]; // Stores the current block of encrypted message

        // Load the current block of encrypted message into state
        for (int i = 0; i < 16; i++) {
            state[i] = encryptedMessage[startIdx + i];
        }

        // Initial round
        cudaInitialRound(state, expandedKey + 160);

        // Perform 9 rounds of decryption
        for (int i = 8; i >= 0; i--) {
            cudaDecRound(state, expandedKey + (16 * (i + 1)));
        }

        // Final round
        cudaSubRoundKey(state, expandedKey);

        // Copy decrypted state to buffer
        for (int i = 0; i < 16; i++) {
            decryptedMessage[startIdx + i] = state[i];
        }
    }
}

string cudaDecrypt(string inputFileName, string inputFileType) {
    std::cout << "=======================================" << std::endl;
    std::cout << " 128-bit AES CUDA 11.6 Decryption Tool " << std::endl;
    std::cout << "=======================================" << std::endl;

    double start, end, total=0;
    double kstart, kend;
    double fullstart = omp_get_wtime();

    // Define chunk size (adjust as needed)
    const size_t chunkSize = 1024 * 1024 * 100; // 500 MB

    // Read in the encrypted message from encrypted file
    std::ifstream infile(inputFileName, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Unable to open encrypted file: " << inputFileName << std::endl;
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
    std::string outputFileName = "decrypted_" + inputFileName.substr(0, inputFileName.find_last_of(".")) + inputFileType;
    std::ofstream outfile(outputFileName, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Unable to open file for writing decrypted message" << std::endl;
        infile.close();
        cudaFree(d_expandedKey);
        return "";
    }


    bool isFirstChunk = true;
    size_t lastChunkSize = 0;

    // Loop through the input file and decrypt each chunk
    while (!infile.eof()) {
        unsigned char* chunk = new unsigned char[chunkSize];
        infile.read(reinterpret_cast<char*>(chunk), chunkSize);
        size_t bytesRead = infile.gcount();

        if (bytesRead == 0) {
            delete[] chunk;
            break;
        }

        unsigned char* decryptedChunk = new unsigned char[bytesRead];

        unsigned char* d_encryptedChunk;
        unsigned char* d_decryptedChunk;
        cudaMalloc((void**)&d_encryptedChunk, bytesRead * sizeof(unsigned char));
        cudaMalloc((void**)&d_decryptedChunk, bytesRead * sizeof(unsigned char));
        cudaMemcpy(d_encryptedChunk, chunk, bytesRead * sizeof(unsigned char), cudaMemcpyHostToDevice);

        int numBlocks = (bytesRead + blockSize - 1) / blockSize;
        start = omp_get_wtime();
        cudaAESDecrypt << <numBlocks, blockSize >> > (d_encryptedChunk, d_expandedKey, d_decryptedChunk, bytesRead);
        end = omp_get_wtime() - start;
        cudaMemcpy(decryptedChunk, d_decryptedChunk, bytesRead * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        if (!infile.eof()) {
            outfile.write(reinterpret_cast<char*>(decryptedChunk), bytesRead);
        }
        else {
            lastChunkSize = bytesRead;
            if (lastChunkSize > 0) {
                int paddingLength = decryptedChunk[lastChunkSize - 1];
                if (paddingLength > 0 && paddingLength <= 16) {
                    outfile.write(reinterpret_cast<char*>(decryptedChunk), lastChunkSize - paddingLength);
                }
                else {
                    outfile.write(reinterpret_cast<char*>(decryptedChunk), lastChunkSize);
                }
            }
        }

        delete[] chunk;
        delete[] decryptedChunk;
        cudaFree(d_encryptedChunk);
        cudaFree(d_decryptedChunk);
        total += end;
    }

    infile.close();
    outfile.close();

    std::cout << "Wrote decrypted message to file: " << outputFileName << std::endl;
    double fullend = omp_get_wtime() - fullstart;
    std::cout << "Time taken to perform AES decryption: " << total << std::endl;
    std::cout << "Total Time taken for the entire process: " << fullend << std::endl;

    // Free memory
    cudaFree(d_expandedKey);
    return outputFileName;
}