#clang -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Xpreprocessor -fopenmp -I/usr/local/include -lomp -L/usr/local/lib simd_example.c -o simd_example
#gcc -O3 -mavx -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Xpreprocessor -fopenmp -I/usr/local/include -lomp -L/usr/local/lib timing_aligned.c -o timing_aligned
#gcc -O3 -mavx -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Xpreprocessor -fopenmp -I/usr/local/include -lomp -L/usr/local/lib avx.c -o avx
clang -O3 -ffast-math -mavx2 -ftime-trace -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Xpreprocessor -fopenmp -I/usr/local/include -lomp -L/usr/local/lib avx_loglike.c -o avx_loglike

#otool -tvV avx_loglike | grep -E 'vmovapd|vaddpd|mul'



#machdep.cpu.features: FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX EST TM2 SSSE3 FMA CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES PCID XSAVE OSXSAVE SEGLIM64 TSCTMR AVX1.0 RDRAND F16C