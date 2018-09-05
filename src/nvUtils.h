/*
* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#ifndef NV_CODEC_UTILS_H
#define NV_CODEC_UTILS_H

#pragma once
#include <chrono>
#include <logger.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

extern simplelogger::Logger *logger;

#ifdef _WIN32
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#else
#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif


#ifdef __cuda_cuda_h__
inline bool CHECK_(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        LOG_ERROR(logger, "CUDA error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#ifdef __CUDA_RUNTIME_H__
inline bool CHECK_(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        LOG_ERROR(logger, "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#ifdef _NV_ENCODEAPI_H_
inline bool CHECK_(NVENCSTATUS e, int iLine, const char *szFile) {
    if (e != NV_ENC_SUCCESS) {
        LOG_ERROR(logger, "NVENC error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#ifdef _WINERROR_
inline bool CHECK_(HRESULT e, int iLine, const char *szFile) {
    if (e != S_OK) {
        LOG_ERROR(logger, "HRESULT error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#if defined(__gl_h_) || defined(__GL_H__)
inline bool CHECK_(GLenum e, int iLine, const char *szFile) {
    if (e != 0) {
        LOG_ERROR(logger, "GLenum error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#define ck(call) CHECK_(call, __LINE__, __FILE__)
/*
*/

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
inline int _getch( ) {
  struct termios oldt, newt;
  int ch;
  tcgetattr( STDIN_FILENO, &oldt );
  newt = oldt;
  newt.c_lflag &= ~( ICANON | ECHO );
  tcsetattr( STDIN_FILENO, TCSANOW, &newt );
  ch = getchar();
  tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
  return ch;
}
#define _stricmp strcasecmp
#endif

class BufferedFileReader {
public:
    BufferedFileReader(const char *szFileName) {
        struct stat st;

        if (stat(szFileName, &st) != 0) {
            return;
        }
        
        nSize = st.st_size;
        pBuf = new uint8_t[nSize];
        if (!pBuf) {
            LOG_ERROR(logger, "Failed to allocate memory in BufferedReader");
            return;
        }

        FILE *fp = fopen(szFileName, "rb");
        int nRead = fread(pBuf, 1, nSize, fp);
        fclose(fp);

        assert(nRead == nSize);
    }
    ~BufferedFileReader() {
        if (pBuf) {
            delete[] pBuf;
        }
    }
    bool GetBuffer(uint8_t **ppBuf, int *pnSize) {
        if (!pBuf) {
            return false;
        }

        *ppBuf = pBuf;
        *pnSize = nSize;
        return true;
    }

private:
    uint8_t *pBuf = NULL;
    int nSize = 0;
};

/*
class YuvConverter {
public:
    YuvConverter(int nWidth, int nHeight) : nWidth(nWidth), nHeight(nHeight) {
        pu = new uint8_t[nWidth * nHeight / 4];
    }
    ~YuvConverter() {
        delete pu;
    }
    void I420ToNv12(uint8_t *pFrame, int nPitch = 0) {
        if (nPitch == 0) {
            nPitch = nWidth;
        }
        uint8_t *puv = pFrame + nPitch * nHeight;
        if (nPitch == nWidth) {
            memcpy(pu, puv, nWidth * nHeight / 4);
        } else {
            for (int i = 0; i < nHeight / 2; i++) {
                memcpy(pu + nWidth / 2 * i, puv + nPitch / 2 * i, nWidth / 2);
            }
        }
        uint8_t *pv = puv + (nPitch / 2) * (nHeight / 2);
        for (int y = 0; y < nHeight / 2; y++) {
            for (int x = 0; x < nWidth / 2; x++) {
                puv[y * nPitch + x * 2] = pu[y * nWidth / 2 + x];
                puv[y * nPitch + x * 2 + 1] = pv[y * nPitch / 2 + x];
            }
        }
    }

private:
    uint8_t *pu;
    int nWidth, nHeight;
};
*/
class StopWatch {
public:
    void Start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    double Stop() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> t0;
};
/*
class StopWatchNew {
public:
    void Start() {
        //t0 = std::chrono::high_resolution_clock::now();
    	gettimeofday(&t0, NULL);
	}
    double Stop() {
    	struct timeval t1;
		gettimeofday(&t1, NULL);
		return (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)/1000000;
	}

private:
	struct timeval t0;
};*/

#endif // NV_CODEC_UTILS_H
