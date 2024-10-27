#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <omp.h> // Include OpenMP for omp_get_wtime()
#include "structures.h"

using namespace std;

/* Used in Round() and serves as the final round during decryption
 * SubRoundKey is simply an XOR of a 128-bit block with the 128-bit key.
 * So basically does the same as AddRoundKey in the encryption
 */
void SubRoundKey(unsigned char* state, unsigned char* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* InverseMixColumns uses mul9, mul11, mul13, mul14 look-up tables
 * Unmixes the columns by reversing the effect of MixColumns in encryption
 */
void InverseMixColumns(unsigned char* state) {
    unsigned char tmp[16];

    tmp[0] = (unsigned char)mul14[state[0]] ^ mul11[state[1]] ^ mul13[state[2]] ^ mul9[state[3]];
    tmp[1] = (unsigned char)mul9[state[0]] ^ mul14[state[1]] ^ mul11[state[2]] ^ mul13[state[3]];
    tmp[2] = (unsigned char)mul13[state[0]] ^ mul9[state[1]] ^ mul14[state[2]] ^ mul11[state[3]];
    tmp[3] = (unsigned char)mul11[state[0]] ^ mul13[state[1]] ^ mul9[state[2]] ^ mul14[state[3]];

    tmp[4] = (unsigned char)mul14[state[4]] ^ mul11[state[5]] ^ mul13[state[6]] ^ mul9[state[7]];
    tmp[5] = (unsigned char)mul9[state[4]] ^ mul14[state[5]] ^ mul11[state[6]] ^ mul13[state[7]];
    tmp[6] = (unsigned char)mul13[state[4]] ^ mul9[state[5]] ^ mul14[state[6]] ^ mul11[state[7]];
    tmp[7] = (unsigned char)mul11[state[4]] ^ mul13[state[5]] ^ mul9[state[6]] ^ mul14[state[7]];

    tmp[8] = (unsigned char)mul14[state[8]] ^ mul11[state[9]] ^ mul13[state[10]] ^ mul9[state[11]];
    tmp[9] = (unsigned char)mul9[state[8]] ^ mul14[state[9]] ^ mul11[state[10]] ^ mul13[state[11]];
    tmp[10] = (unsigned char)mul13[state[8]] ^ mul9[state[9]] ^ mul14[state[10]] ^ mul11[state[11]];
    tmp[11] = (unsigned char)mul11[state[8]] ^ mul13[state[9]] ^ mul9[state[10]] ^ mul14[state[11]];

    tmp[12] = (unsigned char)mul14[state[12]] ^ mul11[state[13]] ^ mul13[state[14]] ^ mul9[state[15]];
    tmp[13] = (unsigned char)mul9[state[12]] ^ mul14[state[13]] ^ mul11[state[14]] ^ mul13[state[15]];
    tmp[14] = (unsigned char)mul13[state[12]] ^ mul9[state[13]] ^ mul14[state[14]] ^ mul11[state[15]];
    tmp[15] = (unsigned char)mul11[state[12]] ^ mul13[state[13]] ^ mul9[state[14]] ^ mul14[state[15]];

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

// Shifts rows right (rather than left) for decryption
void DecShiftRows(unsigned char* state) {
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
void DecSubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) { // Perform substitution to each of the 16 bytes
        state[i] = inv_s[state[i]];
    }
}

/* Each round operates on 128 bits at a time
 * The number of rounds is defined in AESDecrypt()
 * Not surprisingly, the steps are the encryption steps but reversed
 */
void DecRound(unsigned char* state, unsigned char* key) {
    SubRoundKey(state, key);
    InverseMixColumns(state);
    DecShiftRows(state);
    DecSubBytes(state);
}

// Same as Round() but no InverseMixColumns
void InitialRound(unsigned char* state, unsigned char* key) {
    SubRoundKey(state, key);
    DecShiftRows(state);
    DecSubBytes(state);
}

/* The AES decryption function
 * Organizes all the decryption steps into one function
 */
void AESDecrypt(const unsigned char* encryptedMessage, unsigned char* expandedKey, unsigned char* decryptedMessage)
{
    unsigned char state[16]; // Stores the first 16 bytes of encrypted message

    for (int i = 0; i < 16; i++) {
        state[i] = encryptedMessage[i];
    }

    InitialRound(state, expandedKey + 160);

    int numberOfRounds = 9;

    for (int i = 8; i >= 0; i--) {
        DecRound(state, expandedKey + (16 * (i + 1)));
    }

    SubRoundKey(state, expandedKey); // Final round

    // Copy decrypted state to buffer
    for (int i = 0; i < 16; i++) {
        decryptedMessage[i] = state[i];
    }
}

void decrypt2() {
    cout << "=============================" << endl;
    cout << " 128-bit AES Decryption Tool " << endl;
    cout << "=============================" << endl;

    double start, end;

    double kstart, kend;

    // Read in the encrypted message from message.aes
    string encryptedStr;
    ifstream infile("file.aes", ios::binary);
    if (!infile.is_open()) {
        cout << "Unable to open file message.aes" << endl;
        return;
    }

    stringstream buffer;
    buffer << infile.rdbuf();
    encryptedStr = buffer.str();
    infile.close();

    // Convert the encrypted message to unsigned char array
    unsigned char* encryptedMessage = new unsigned char[encryptedStr.length()];
    for (size_t i = 0; i < encryptedStr.length(); ++i) {
        encryptedMessage[i] = static_cast<unsigned char>(encryptedStr[i]);
    }

    // Read in the key from keyfile
    string keyStr;
    ifstream keyfile("keyfile", ios::binary);
    if (!keyfile.is_open()) {
        cout << "Unable to open file keyfile" << endl;
        delete[] encryptedMessage; // Free allocated memory before return
        return;
    }

    getline(keyfile, keyStr);
    keyfile.close();

    // Convert the key to unsigned char array
    unsigned char key[16];
    istringstream hexCharsStream(keyStr);
    for (int i = 0; i < 16; ++i) {
        int c;
        hexCharsStream >> hex >> c;
        key[i] = static_cast<unsigned char>(c);
    }

    // Expand the key
    unsigned char expandedKey[176];

    kstart = omp_get_wtime();
    KeyExpansion(key, expandedKey); // Assume KeyExpansion function is defined elsewhere
    kend = omp_get_wtime() - kstart;

    cout << "Time taken for key expansion: " << kend << endl;

    // Decrypt the message
    int messageLen = encryptedStr.length();
    unsigned char* decryptedMessage = new unsigned char[messageLen + 1]; // Allocate space for null terminator

    start = omp_get_wtime();
    for (int i = 0; i < messageLen; i += 16) {
        AESDecrypt(encryptedMessage + i, expandedKey, decryptedMessage + i);
    }
    end = omp_get_wtime() - start;

    int lastBlockIndex = (messageLen / 16) * 16;
    decryptedMessage[lastBlockIndex] = '\0'; // Null terminate the decrypted message

    // Write the decrypted message to a new file
    ofstream outfile("decrypted_file.mp4", ios::out | ios::binary);
    if (!outfile.is_open()) {
        cout << "Unable to open file for writing decrypted message" << endl;
        delete[] encryptedMessage;
        delete[] decryptedMessage;
        return;
    }

    outfile.write(reinterpret_cast<char*>(decryptedMessage), lastBlockIndex);
    outfile.close();
    cout << "Decrypted message written to decrypted_message.txt" << endl;
    cout << "Time taken to perform AES decryption: " << end << endl;

    // Free memory
    delete[] encryptedMessage;
    delete[] decryptedMessage;
}

string decrypt(string inputFileName, string inputFileType) {
    std::cout << "=============================" << std::endl;
    std::cout << " 128-bit AES Decryption Tool " << std::endl;
    std::cout << "=============================" << std::endl;

    double start, end, total = 0;
    double kstart, kend;

    // Open the input file
    std::ifstream infile(inputFileName, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Unable to open encrypted file: " << inputFileName << std::endl;
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
    std::string outputFileName = "decrypted_" + inputFileName.substr(0, inputFileName.find_last_of(".")) + inputFileType;
    std::ofstream outfile(outputFileName, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Unable to open file for writing decrypted message" << std::endl;
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
                    AESDecrypt(inputBuffer + i, expandedKey, inputBuffer + i); // Overwrite inputBuffer with encrypted data
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

    std::cout << "Wrote decrypted message to file: " << outputFileName << std::endl;
    std::cout << "Time taken to perform AES decryption: " << total << std::endl;
    return outputFileName;
}