# COM4521/COM6521 Assignment Code Review & Lecture Compliance Report

> 작성일: 2026-05-17  
> 대상: `openmp.c`, `cuda.cu` (기존 제출 코드)  
> 참고 Lecture Slides: 04b, 05, 06a, 06b, 07a, 07b, 08a, 08b, lab04~07

---

## 1. Executive Summary

현재 제출 코드는 **Shared Memory Tiling**, **Constant Memory**, **Coalesced Access**, **False Sharing Avoidance** 등 COM4521/COM6521에서 가르친 핵심 기법을 대부분 적용했습니다. 그러나 **Lecture 07a의 Warp Shuffle**, **Lecture 06a의 Bank Conflict 회피**, **OpenMP의 Nested Parallel Region 제거**, 그리고 **Profiling Instrumentation** 측면에서 추가 개선의 여지가 큽니다. 본 문서는 현재 코드와 Lecture/Methodology 간의 차이를 정밀 비교하고, **For Loop 최소화의 타당성**을 분석하며, **Profiling 전략**을 제시합니다.

---

## 2. 현재 코드의 강점 (Lecture와 일치하는 부분)

| 기법 | Lecture | 현재 코드 적용 여부 | 평가 |
|------|---------|---------------------|------|
| **Shared Memory Tiling + Halo** | 06a, lab06 | CountGliders, Emboss에 적용 | ✅ 우수 |
| **Constant Memory** | 05, lab05 | `d_glider_masks[16]`, `d_emboss_kernel[3][3]` | ✅ 우수 |
| **Coalesced Global Memory Access** | 06b | Consecutive threads가 consecutive data 접근 | ✅ 우수 |
| **Grid-Stride Loop** | 04b, 07b | `histogram_kernel`에서 stride 루프 사용 | ✅ 우수 |
| **False Sharing Avoidance** | 일반 병렬 컴퓨팅 | OpenMP Histogram에 Padding 적용 | ✅ 우수 |
| **Parallel Reduction** | 07b | CountGliders에서 block-level reduction 사용 | ⚠️ 부분적 (개선 필요) |
| **Atomics to Shared then Global** | 07a, lab07 | Histogram에서 local shared atomic 후 global atomic | ✅ 우수 |
| **Block Size = 256 (32의 배수)** | 07a | 16×16 블록 사용 | ✅ 우수 |

---

## 3. 부족한 점 / 틀린 점 (Lecture와 차이나는 부분)

### 3.1 CUDA CountGliders — Shared Memory Reduction의 Bank Conflict ⚠️ **중요**

- **Lecture 참조**: 06a (Bank Conflicts), 07a (Warp Shuffle), 07b (Reduction)
- **현재 코드**:
  ```cuda
  for (int stride = ((int)blockDim.x * (int)blockDim.y) / 2; stride > 0; stride >>= 1) {
      if (local_idx < stride) {
          block_count[local_idx] += block_count[local_idx + stride];
      }
      __syncthreads();
  }
  ```
- **문제**: `unsigned int` 배열(4 bytes)에서 `local_idx`와 `local_idx + stride`가 **같은 Shared Memory Bank**에 매핑될 수 있습니다. Shared Memory는 32개 bank이고, 각 bank는 4 bytes wide입니다. `stride`가 128, 64, 32, 16, 8, 4, 2일 때:
  - 예: `stride = 16` → offset = 64 bytes = 16 banks. `local_idx=0`은 bank 0과 bank 0에 동시 접근 → **2-way bank conflict**.
  - `stride = 8` → 4-way conflict.
- **수정안**: Lecture 07a에서 배운 **`__shfl_down_sync`**를 사용하여 warp-level reduction을 수행하고, warp 간 reduction만 shared memory(32개 entry)로 처리합니다. 이는 bank conflict를 완전히 제거하고 shared memory 사용량도 줄입니다.

### 3.2 OpenMP Histogram — Nested Parallel Regions ⚠️ **중요**

- **Lecture 참조**: 일반 OpenMP Best Practice (Thread 생성 overhead)
- **현재 코드**:
  ```c
  #pragma omp parallel
  {
      // ... local hist 계산 ...
      #pragma omp parallel for private(t) schedule(static)
      for (b = 0; b < HISTOGRAM_LEN; ++b) { ... }
  }
  ```
- **문제**: 외부 `omp parallel` 안에 **또 다른 `omp parallel for`**가 있습니다. 이는 새로운 thread team을 생성하므로 overhead가 2배로 발생합니다. `omp_get_nested()`가 켜져 있지 않으면 실제 중첩은 안 될 수 있지만, 명시적으로 중첩을 요청하는 코드는 지양해야 합니다.
- **수정안**: **단일 `omp parallel` region** 내에서 `omp for`를 두 번 사용합니다.

### 3.3 CUDA Kernel Signatures — Missing `__restrict__` ⚠️

- **Lecture 참조**: 05 (Read-only / Texture Cache), 06b (Memory Coalescing)
- **문제**: Kernel parameter로 전달되는 read-only pointer(`cells`, `numbers`, `pixels`)에 `__restrict__` qualifier가 없습니다. `const ... __restrict__`를 추가하면 compiler가 `__ldg()` (Load Global through Read-only cache)를 사용할 수 있어 L2 cache hit rate가 향상될 수 있습니다.
- **수정안**: `const unsigned char* __restrict__ cells` 등으로 변경.

### 3.4 Warp-Level Primitives 미사용 ⚠️

- **Lecture 참조**: 07a (Warp Level CUDA & Atomics)
- **문제**: `__shfl_down_sync`, `__ballot_sync`, `__popc` 등 warp-level primitive를 전혀 사용하지 않았습니다. 특히 reduction과 voting 패턴에서 매우 효과적입니다.
- **수정안**: CountGliders의 block reduction 대신 warp shuffle reduction 도입.

### 3.5 Profiling Instrumentation 부재 ⚠️

- **Lecture 참조**: 08a (Profiling), 08b (Nsight)
- **문제**: 코드 내에 **정밀 타이밍 코드**(`cudaEvent_t`, `omp_get_wtime()`)가 없습니다. Report에서 profiling을 논하려면 수치 근거가 필요합니다.
- **수정안**: Host wrapper 함수에 임시로 `cudaEvent_t`를 삽입하거나, 외부 profiler 명령어를 사용합니다.

### 3.6 OpenMP CountGliders — `build_glider_masks` 반복 호출

- **문제**: Pattern은 static constant인데, `openmp_countGliders()`가 호출될 때마다 `build_glider_masks()`를 실행합니다. 큰 overhead는 아니나, 완성도 측면에서 `static const`로 미리 계산해 두는 것이 바람직합니다.
- **수정안**: File-scope `static const unsigned short GLIDER_MASKS[16]`로 선언.

### 3.7 OpenMP Histogram — 불필요한 `private(t)`

- **문제**: `#pragma omp parallel for private(t)`에서 `t`는 이미 inner loop 변수로 **자동 private**입니다. 불필요한 절입니다.
- **수정안**: `private(t)` 제거.

### 3.8 OpenMP/CUDA CountGliders — 16-way Mask 비교 Loop

- **현재**: `for (int i = 0; i < 16; ++i)`로 16개 mask를 순차 비교.
- **분석**: 16개는 작은 수이므로 unroll pragma(`#pragma unroll`)를 사용하면 compiler가 전개할 수 있습니다. 현재는 compiler dependent입니다.
- **수정안**: CUDA 버전에서는 `#pragma unroll` 추가 권장.

---

## 4. For Loop 최소화 이유 및 Profiling 가이드

### 4.1 왜 For Loop을 최소화했는가?

현재 코드의 `load_window_mask` 함수에서 3×3 window(9개 픽셀)를 9개의 `if`문으로 직접 전개(unroll)했습니다. 이에 대한 이유는 다음과 같습니다:

1. **Loop Overhead 제거**  
   고정된 9번 반복에서는 loop control instruction(`inc`, `cmp`, `branch`)이 9×3=27개 이상 추가됩니다.数百万~数十억 개의 window를 처리할 때 이 overhead가 누적되어 유의미한 차이를 만듭니다.

2. **Branch Prediction Penalty 감소**  
   작은 loop는 branch predictor가 루프를 정확히 예측하지만, loop exit 시 misprediction penalty(10~20 cycles)가 발생합니다. Unroll하면 loop branch 자체가 사라집니다.

3. **SIMD / ILP (Instruction Level Parallelism)**  
   OpenMP에서 compiler는 unrolled code를 더 쉽게 **auto-vectorize**할 수 있습니다. CUDA에서는 9개의 독립적인 bitwise OR 연산이 instruction scheduler에 의해 병렬 배치될 수 있어 **ILP**가 향상됩니다.

4. **CUDA Local Memory Spill 방지**  
   CUDA에서 작은 배열에 대한 loop indexing은 non-constant index로 인식될 수 있어 **Local Memory** spill을 유발합니다(05강). Unroll 시 모든 index가 compile-time constant가 되어 **Register-only** access를 보장합니다.

5. **Report에서의 Justification**  
   위 이론적 근거에 더해, 실제 **Profiling 결과**를 첨부하면 강력한 근거가 됩니다. 아래 4.2절의 실험 설계를 따라 측정하세요.

### 4.2 For Loop vs Unroll Profiling 실험 설계

**비교 대상**: `load_window_mask`의 **For Loop 버전** vs **Unroll 버전**

#### OpenMP
- **Metric**: Wall-clock time (`omp_get_wtime()`)
- **Input**: 큰 이미지 (e.g., 4096×4096 random binary)
- **반복**: 100회 평균 (Release mode, `--bench`)
- **예상 결과**: Unroll 버전이 5~15% 빠를 수 있음 (compiler 최적화 수준에 따라 다름)

#### CUDA
- **Metric**: Kernel execution time (`cudaEvent_t`)
- **추가 Metric** (Nsight Compute):
  ```bash
  ncu --metrics sm__cycles_elapsed.avg,\
               l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
               l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
               sm__warps_active.avg.pct_of_peak_sustained_elapsed \
               ./assignment CUDA CG large_input.png
  ```
- **예상 결과**: Unroll 버전이 instruction 수는 늘지만, loop control으로 인한 stall(`stall_not_selected`)이 감소하여 총 cycle이 줄어듦.

---

## 5. 개선 우선순위 및 Report 체크리스트

| 우선순위 | 항목 | 이유 |
|----------|------|------|
| 🔴 P0 | CUDA Reduction → Warp Shuffle | Bank conflict 제거, Lecture 07a 직접 적용 |
| 🔴 P0 | OpenMP Nested Parallelism 제거 | 성능/정확성에 직접 영향 |
| 🟡 P1 | `__restrict__` 추가 | Lecture 05/06b 준수, compiler hint |
| 🟡 P1 | Profiling 코드 삽입 | Report 수치 근거 필수 |
| 🟢 P2 | `build_glider_masks` static const | 미세 최적화 |
| 🟢 P2 | `private(t)` 제거 | 코드 품질 |

**Marking Criteria 대비**:
- [ ] **Terminology**: Shared memory, bank conflict, warp shuffle, occupancy, coalescing, false sharing 등 정확한 용어 사용
- [ ] **Knowledge**: Lecture에서 배운 기법(상기 8개)을 모두 언급하고 적용 이유 설명
- [ ] **Understanding**: Bottleneck 분석 → Profiling 수치 → Optimization 순서로 논증
- [ ] **Communication**: Figure 포함 (Shared Memory tiling diagram, Warp shuffle diagram)
- [ ] **Correctness**: 5개 테스트 이미지 통과 (marking criteria 참조)

---

## 6. 참고 파일

- `src/openmp_loop_version.c`: `load_window_mask`에 for loop을 사용한 비교용 버전
- `src/cuda_loop_version.cu`: `load_window_mask`에 for loop을 사용한 비교용 버전
- `src/openmp_lecture_compliant.c`: Lecture 기법을 반영한 개선 OpenMP 코드
- `src/cuda_lecture_compliant.cu`: Lecture 기법을 반영한 개선 CUDA 코드
- `PROFILE_GUIDE.md`: OpenMP/CUDA profiling 명령어 및 스크립트

---

*본 문서는 COM4521/COM6521 Assignment Marking Criteria와 Lecture Slides(04b~09b, lab04~07)를 기반으로 작성되었습니다.*
