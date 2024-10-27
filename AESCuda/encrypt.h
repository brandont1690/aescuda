#pragma once
#include <omp.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include "structures.h"

using namespace std;

/* Serves as the initial round during encryption
 * AddRoundKey is simply an XOR of a 128-bit block with the 128-bit key.
 */
void AddRoundKey(unsigned char* state, unsigned char* roundKey) {
	for (int i = 0; i < 16; i++) {
		state[i] ^= roundKey[i];
	}
}

/* Perform substitution to each of the 16 bytes
 * Uses S-box as lookup table
 */
void SubBytes(unsigned char* state) {
	for (int i = 0; i < 16; i++) {
		state[i] = s[state[i]];
	}
}

// Shift left, adds diffusion
void ShiftRows(unsigned char* state) {
	unsigned char tmp[16];

	/* Column 1 */
	tmp[0] = state[0];
	tmp[1] = state[5];
	tmp[2] = state[10];
	tmp[3] = state[15];

	/* Column 2 */
	tmp[4] = state[4];
	tmp[5] = state[9];
	tmp[6] = state[14];
	tmp[7] = state[3];

	/* Column 3 */
	tmp[8] = state[8];
	tmp[9] = state[13];
	tmp[10] = state[2];
	tmp[11] = state[7];

	/* Column 4 */
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
void MixColumns(unsigned char* state) {
	unsigned char tmp[16];

	tmp[0] = (unsigned char)mul2[state[0]] ^ mul3[state[1]] ^ state[2] ^ state[3];
	tmp[1] = (unsigned char)state[0] ^ mul2[state[1]] ^ mul3[state[2]] ^ state[3];
	tmp[2] = (unsigned char)state[0] ^ state[1] ^ mul2[state[2]] ^ mul3[state[3]];
	tmp[3] = (unsigned char)mul3[state[0]] ^ state[1] ^ state[2] ^ mul2[state[3]];

	tmp[4] = (unsigned char)mul2[state[4]] ^ mul3[state[5]] ^ state[6] ^ state[7];
	tmp[5] = (unsigned char)state[4] ^ mul2[state[5]] ^ mul3[state[6]] ^ state[7];
	tmp[6] = (unsigned char)state[4] ^ state[5] ^ mul2[state[6]] ^ mul3[state[7]];
	tmp[7] = (unsigned char)mul3[state[4]] ^ state[5] ^ state[6] ^ mul2[state[7]];

	tmp[8] = (unsigned char)mul2[state[8]] ^ mul3[state[9]] ^ state[10] ^ state[11];
	tmp[9] = (unsigned char)state[8] ^ mul2[state[9]] ^ mul3[state[10]] ^ state[11];
	tmp[10] = (unsigned char)state[8] ^ state[9] ^ mul2[state[10]] ^ mul3[state[11]];
	tmp[11] = (unsigned char)mul3[state[8]] ^ state[9] ^ state[10] ^ mul2[state[11]];

	tmp[12] = (unsigned char)mul2[state[12]] ^ mul3[state[13]] ^ state[14] ^ state[15];
	tmp[13] = (unsigned char)state[12] ^ mul2[state[13]] ^ mul3[state[14]] ^ state[15];
	tmp[14] = (unsigned char)state[12] ^ state[13] ^ mul2[state[14]] ^ mul3[state[15]];
	tmp[15] = (unsigned char)mul3[state[12]] ^ state[13] ^ state[14] ^ mul2[state[15]];

	for (int i = 0; i < 16; i++) {
		state[i] = tmp[i];
	}
}

/* Each round operates on 128 bits at a time
 * The number of rounds is defined in AESEncrypt()
 */
void Round(unsigned char* state, unsigned char* key) {
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, key);
}

// Same as Round() except it doesn't mix columns
void FinalRound(unsigned char* state, unsigned char* key) {
	SubBytes(state);
	ShiftRows(state);
	AddRoundKey(state, key);
}

/* The AES encryption function
 * Organizes the confusion and diffusion steps into one function
 */
void AESEncrypt(unsigned char* message, unsigned char* expandedKey, unsigned char* encryptedMessage) {
	unsigned char state[16]; // Stores the first 16 bytes of original message

	for (int i = 0; i < 16; i++) {
		state[i] = message[i];
	}

	int numberOfRounds = 9;

	AddRoundKey(state, expandedKey); // Initial round

	for (int i = 0; i < numberOfRounds; i++) {
		Round(state, expandedKey + (16 * (i + 1)));
	}

	FinalRound(state, expandedKey + 160);

	// Copy encrypted state to buffer
	for (int i = 0; i < 16; i++) {
		encryptedMessage[i] = state[i];
	}
}

string encrypt(string inputFileName) {
	std::cout << "=============================" << std::endl;
	std::cout << " 128-bit AES Encryption Tool " << std::endl;
	std::cout << "=============================" << std::endl;

	double start, end, total = 0;
	double kstart, kend;

	// Open the input file
	std::ifstream infile(inputFileName, std::ios::in | std::ios::binary);
	if (!infile.is_open()) {
		std::cout << "Unable to open input file: " << inputFileName << std::endl;
		return "";
	}

	// Open the key file and perform key expansion
	std::ifstream keyfile("keyfile", std::ios::in | std::ios::binary);
	if (!keyfile.is_open()) {
		std::cout << "Unable to open key file" << std::endl;
		infile.close();
		return "";
	}

	std::string str;
	getline(keyfile, str); // The first line of file should be the key
	keyfile.close();

	std::istringstream hex_chars_stream(str);
	unsigned char key[16];
	int i = 0;
	unsigned int c;
	while (hex_chars_stream >> std::hex >> c) {
		key[i] = static_cast<unsigned char>(c);
		i++;
	}

	unsigned char expandedKey[176];
	kstart = omp_get_wtime();
	KeyExpansion(key, expandedKey);
	kend = omp_get_wtime() - kstart;
	std::cout << "Time taken for key expansion: " << kend << std::endl;

	// Open the output file for writing encrypted content
	std::string outputFileName = inputFileName.substr(0, inputFileName.find_last_of(".")) + ".aes";
	std::ofstream outfile(outputFileName, std::ios::out | std::ios::binary);
	if (!outfile.is_open()) {
		std::cout << "Unable to open output file: " << outputFileName << std::endl;
		infile.close();
		return "";
	}

	// Define chunk size
	const size_t chunkSize = 1024 * 1024 * 100; // 100 MB

	// Encryption
	while (!infile.eof()) {
		unsigned char* inputBuffer = new unsigned char[chunkSize];
		infile.read(reinterpret_cast<char*>(inputBuffer), chunkSize);
		size_t bytesRead = infile.gcount();
		if (bytesRead > 0) {
			start = omp_get_wtime();
			for (size_t i = 0; i < bytesRead; i += 16) {
				// Ensure inputBuffer index is within bounds
				if (i + 16 <= bytesRead) {
					AESEncrypt(inputBuffer + i, expandedKey, inputBuffer + i); // Overwrite inputBuffer with encrypted data
				}
			}
			end = omp_get_wtime() - start;
			total += end;
			outfile.write(reinterpret_cast<char*>(inputBuffer), bytesRead);
		}
		delete[] inputBuffer;
	}

	// Close files
	infile.close();
	outfile.close();

	std::cout << "Wrote encrypted message to file: " << outputFileName << std::endl;
	std::cout << "Time taken to perform AES encryption: " << total << std::endl;

	return outputFileName;
}


void encrypt2() {

	cout << "=============================" << endl;
	cout << " 128-bit AES Encryption Tool   " << endl;
	cout << "=============================" << endl;

	double start, end;
	double kstart, kend;

	string filename = "file.mp4";

	// Open the input file
	ifstream infile(filename, ios::in | ios::binary);
	if (!infile.is_open()) {
		cout << "Unable to open input file: " << filename << endl;
		return;
	}

	// Get the length of the file
	infile.seekg(0, ios::end);
	int fileLength = infile.tellg();
	infile.seekg(0, ios::beg);

	// Allocate memory to hold the file content
	unsigned char* fileContent = new unsigned char[fileLength];

	// Read the file content into the buffer
	infile.read(reinterpret_cast<char*>(fileContent), fileLength);

	// Close the input file
	infile.close();

	// Perform padding if necessary

	// Key expansion
	string str;
	ifstream keyfile("keyfile", ios::in | ios::binary);
	if (!keyfile.is_open()) {
		cout << "Unable to open key file" << endl;
		delete[] fileContent;
		return;
	}

	getline(keyfile, str); // The first line of file should be the key
	keyfile.close();

	istringstream hex_chars_stream(str);
	unsigned char key[16];
	int i = 0;
	unsigned int c;
	while (hex_chars_stream >> hex >> c)
	{
		key[i] = c;
		i++;
	}

	unsigned char expandedKey[176];

	kstart = omp_get_wtime();
	KeyExpansion(key, expandedKey); // Assume KeyExpansion function is defined elsewhere
	kend = omp_get_wtime() - kstart;

	cout << "Time taken for key expansion: " << kend << endl;

	// Encryption
	unsigned char* encryptedContent = new unsigned char[fileLength];

	start = omp_get_wtime();
	for (int i = 0; i < fileLength; i += 16) {
		AESEncrypt(fileContent + i, expandedKey, encryptedContent + i);
	}
	end = omp_get_wtime() - start;

	// Write the encrypted content to output file
	string outputFilename = filename.substr(0, filename.find_last_of(".")) + ".aes";
	ofstream outfile(outputFilename, ios::out | ios::binary);
	if (!outfile.is_open()) {
		cout << "Unable to open output file: " << outputFilename << endl;
		delete[] fileContent;
		delete[] encryptedContent;
		return;
	}

	outfile.write(reinterpret_cast<char*>(encryptedContent), fileLength);
	outfile.close();
	cout << "Wrote encrypted message to file: " << outputFilename << endl;
	cout << "Time taken to perform AES encryption: " << end << endl;

	// Free memory
	delete[] fileContent;
	delete[] encryptedContent;
}

