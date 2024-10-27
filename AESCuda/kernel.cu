// DPSCAESASGM.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/* encrypt.cpp
 * Performs encryption using AES 128-bit
 * @author Cecelia Wisniewska
 */
#include <omp.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include "structures.h"
#include "encrypt.h"
#include "encrypt.cuh"
#include "testCudaEncrypt.cuh"
//#include "testCudaDecrypt.cuh"
#include "decrypt.h"
#include "decrypt.cuh"

bool compareFiles(const std::string& file1, const std::string& file2, int bytesToCompare) {
    std::ifstream infile1(file1, std::ios::binary);
    std::ifstream infile2(file2, std::ios::binary);

    if (!infile1 || !infile2) {
        std::cerr << "Error: Unable to open files for comparison." << std::endl;
        return false;
    }

    infile1.seekg(0, std::ios::end);
    infile2.seekg(0, std::ios::end);
    std::streampos size1 = infile1.tellg();
    std::streampos size2 = infile2.tellg();
    infile1.seekg(0, std::ios::beg);
    infile2.seekg(0, std::ios::beg);

    if (std::abs(size1 - size2) > 0) {
        std::cerr << "Error: File sizes are different." << std::endl;
        return false;
    }
    std::cout << "Files sizes are the same." << std::endl;
    // Reset file pointers to the beginning
    infile1.seekg(0, std::ios::beg);
    infile2.seekg(0, std::ios::beg);

    // Determine the actual number of bytes to compare
    int actualBytesToCompare;

    if (std::abs(size1 - size1 + size1) > bytesToCompare) {
        actualBytesToCompare = bytesToCompare;
    }
    else if (std::abs(size1 - size1 + size1) < bytesToCompare) {
        infile1.seekg(0, std::ios::end);
        infile2.seekg(0, std::ios::end);
        int fileSize1 = infile1.tellg();
        int fileSize2 = infile2.tellg();

        actualBytesToCompare = std::min(fileSize1, fileSize2);
    }

    char* buffer1 = new char[actualBytesToCompare];
    char* buffer2 = new char[actualBytesToCompare];

    infile1.read(buffer1, actualBytesToCompare);
    infile2.read(buffer2, actualBytesToCompare);

    bool filesMatch = (infile1.gcount() == infile2.gcount()) && (memcmp(buffer1, buffer2, actualBytesToCompare) == 0);

    delete[] buffer1;
    delete[] buffer2;

    if (filesMatch) {
        std::cout << "Files content match." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Files content differ." << std::endl;
        return false;
    }
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string inputFileName = argv[1];

    std::string inputFileType = inputFileName.substr(inputFileName.find_last_of("."), inputFileName.length());

    //string encryptedFile = encrypt(inputFileName);
	//string decryptedFile = decrypt(encryptedFile, inputFileType);

	string encryptedFile = cudaEncrypt(inputFileName);

    
	string decryptedFile = cudaDecrypt(encryptedFile, inputFileType);

    // Test if the original file matches the decrypted file
    bool filesMatch = compareFiles(inputFileName, decryptedFile, 1024 * 1024 * 100);
    
    // Output the test result
    if (filesMatch) {
        cout << endl << "The original file matches the decrypted file." << endl;
    }
    else {
        cout << endl << "The original file does not match the decrypted file." << endl;
    }

	return 0;
}