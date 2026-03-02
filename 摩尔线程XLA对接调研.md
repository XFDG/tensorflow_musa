# 摩尔线程对接 OpenXLA 支持 TensorFlow XLA 调研

## 1. 执行摘要 (Executive Summary)

### 可行性结论

**摩尔线程对接 OpenXLA 支持 TensorFlow XLA 是完全可行的。**

基于摩尔线程 GPU 高度兼容 CUDA 生态（通过 MUSA 兼容层实现）的特性，技术路径如下：

| 层级         | 责任方             | 工作内容                               |
| ------------ | ------------------ | -------------------------------------- |
| **编译时**   | OpenXLA (Google)   | HLO 优化、LLVM IR 生成（硬件无关部分） |
| **代码生成** | 摩尔线程 + OpenXLA | LLVM IR → MUSA 可执行代码              |
| **运行时**   | 摩尔线程           | PJRT Plugin、MUSA StreamExecutor       |

### 核心架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              编译时 (Compile Time)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   TensorFlow Graph → StableHLO → XLA HLO → LLVM IR → MUSA PTX/二进制             │
│                                                                                 │
│   • 由 OpenXLA Compiler 处理                                                     │
│   • 摩尔线程需确保 LLVM 能生成 MUSA 可执行代码                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          ↓
                                    编译产物
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              运行时 (Runtime)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                    摩尔线程 PJRT Plugin (.so)                             │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐   │   │
│   │  │   Client    │  │   Buffer    │  │       LoadedExecutable          │   │   │
│   │  │  设备管理    │  │  内存管理    │  │    加载编译产物并执行               │   │   │
│   │  └──────┬──────┘  └──────┬──────┘  └───────────────┬─────────────────┘   │   │
│   │         └─────────────────┼─────────────────────────┘                    │   │
│   │                           ▼                                              │   │
│   │              ┌─────────────────────────────┐                             │   │
│   │              │   MUSA StreamExecutor       │                             │   │
│   │              │  (设备抽象层，调用 MUSA API) │                               │   │
│   │              └──────────────┬──────────────┘                             │   │
│   │                             ▼                                            │   │
│   │              ┌─────────────────────────────┐                             │   │
│   │              │   MUSA Runtime/Driver       │                             │   │
│   │              │   (libmusa.so)              │                             │   │
│   │              └──────────────┬──────────────┘                             │   │
│   │                             ▼                                            │   │
│   │              ┌─────────────────────────────┐                             │   │
│   │              │        MUSA GPU             │                             │   │
│   │              └─────────────────────────────┘                             │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 依赖项与架构支持 (Dependencies & Architecture)

### 2.1 前端支持

| 组件       | 版本要求  | 说明                     |
| ---------- | --------- | ------------------------ |
| TensorFlow | 2.15+     | 需要支持 PJRT 的 TF 版本 |
| JAX/Jaxlib | 0.4.30+   | 用于验证和测试（推荐）   |
| Python     | 3.10-3.12 | 与 TensorFlow/JAX 兼容   |

**XLA 启用方式**：

```python
import tensorflow as tf

# 方式1: 自动聚类
tf.config.optimizer.set_jit(True)

# 方式2: 显式 XLA 编译
@tf.function(jit_compile=True)
def xla_func(x):
    return tf.nn.relu(tf.matmul(x, x))
```

### 2.2 后端支持

#### 2.2.1 OpenXLA PJRT 接口分析

PJRT（Pretty much Just another RunTime）是 OpenXLA 提供的**运行时接口**，通过 C API 暴露给上层框架。

**核心接口定义**（`pjrt_c_api.h`）：

```c
// 插件入口函数 - 必须导出
const PJRT_Api* GetPjrtApi(void);

// 关键接口结构（运行时职责）
typedef struct PJRT_Api {
  PJRT_Api_Version api_version;
  
  // 设备管理
  PJRT_Client_Create* Client_Create;
  PJRT_Client_Destroy* Client_Destroy;
  PJRT_Client_Devices* Client_Devices;
  
  // 内存管理
  PJRT_Buffer_Create* Buffer_Create;
  PJRT_Buffer_ToHostBuffer* Buffer_ToHostBuffer;
  
  // 执行（加载已编译的程序）
  PJRT_LoadedExecutable_Execute* LoadedExecutable_Execute;
  
  // 注：编译由 OpenXLA 处理，PJRT 只负责加载执行产物
} PJRT_Api;
```

**重要澄清**：

- ✅ PJRT Plugin 负责：**运行时**的设备管理、内存管理、执行调度
- ❌ PJRT Plugin 不负责：**编译时**的 HLO 优化、LLVM IR 生成

#### 2.2.2 摩尔线程 Backend 定位

| 层级       | 组件                  | 摩尔线程工作                          |
| ---------- | --------------------- | ------------------------------------- |
| **编译时** | HLO → LLVM IR         | 无需修改，OpenXLA 处理                |
| **编译时** | LLVM IR → MUSA 二进制 | 需提供 LLVM MUSA Target 或 PTX 兼容层 |
| **运行时** | PJRT Plugin           | 需完整实现                            |
| **运行时** | StreamExecutor        | 需实现 MUSA StreamExecutor            |
| **运行时** | Kernel 加载执行       | 需调用 MUSA Driver API                |

### 2.3 关键依赖库列表

```
# OpenXLA 依赖（无需修改）
openxla/xla
├── xla/pjrt/c              # PJRT C API 定义
├── xla/service/gpu         # GPU 编译器后端（复用）
├── xla/stream_executor     # StreamExecutor 框架
└── xla/tools/pip_package   # 打包工具

# 摩尔线程软件栈（需要提供）
MUSA SDK
├── libmusa.so              # MUSA Runtime
├── libmusa_driver.so       # MUSA Driver
├── include/musa_runtime_api.h
├── include/musa_driver_api.h
├── muBLAS                  # BLAS 库
├── muFFT                   # FFT 库
└── mcc (可选)               # MUSA 编译器
```

---

## 3. HLO 获取与编译流程 (HLO Pipeline)

### 3.1 编译流程全览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              编译时 (Compile Time)                               │
│                     责任方: OpenXLA Compiler + 摩尔线程                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │ TF GraphDef │ → │ StableHLO   │ → │ XLA HLO     │ → │ LLVM IR     │          │
│  │             │   │ (MLIR)      │   │ (优化后)     │   │ (Generic)   │          │
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────┬──────┘          │
│                                                               │                 │
│                                                               ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    代码生成 (Code Generation)                             │   │
│  │                                                                          │   │
│  │   LLVM IR ──────────────────────────────────────────────→ MUSA PTX/二进制 │   │
│  │        ↑                                                                 │   │
│  │        │ 摩尔线程需要确保:                                                 │    │
│  │        │ • LLVM 有 MUSA Target (或复用 NVPTX)                             │   │
│  │        │ • 生成的代码可在 MUSA GPU 上运行                                   │   │
│  │        │                                                                 │   │
│  │                                                                          │   │
│  │   当前方案: MUSA 有自己的指令集                                              │   │
│  │        → 需提供 LLVM MUSA Backend                                         │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          ↓
                                    编译产物 (Executable)
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              运行时 (Runtime)                                    │
│                         责任方: 摩尔线程 PJRT Plugin                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. 加载 Executable ────────────────────────────────────────────────────────    │
│      PJRT_LoadedExecutable_Create (读取编译产物)                                  │
│                                                                                 │
│   2. 准备输入 Buffer ────────────────────────────────────────────────────────    │
│      PJRT_Buffer_Create (调用 MUSA malloc)                                       │
│                                                                                 │
│   3. 执行 Kernel ────────────────────────────────────────────────────────────    │
│      PJRT_LoadedExecutable_Execute → 调用 MUSA Driver 加载并执行                  │
│                                                                                 │
│   4. 获取结果 ───────────────────────────────────────────────────────────────    │
│      PJRT_Buffer_ToHostBuffer (D2H 拷贝)                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 摩尔线程在编译时的介入点

```cpp
// xla/service/gpu/gpu_compiler.cc
// OpenXLA 代码，摩尔线程无需修改

StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  
  // 1. HLO 优化（硬件无关）
  TF_ASSIGN_OR_RETURN(module, RunHloPasses(std::move(module), ...));
  
  // 2. LLVM IR 生成（硬件无关）
  TF_ASSIGN_OR_RETURN(auto llvm_module, 
                      CompileModuleToLlvmIr(module.get(), ...));
  
  // 3. 代码生成（硬件相关）← 摩尔线程介入点
  // 调用 StreamExecutor::CompileToBinary
  TF_ASSIGN_OR_RETURN(std::string binary,
                      stream_exec->CompileToBinary(llvm_module.get()));
  
  return CreateExecutable(binary, module.get());
}
```

**软件团队需要实现**：

```cpp
// xla/stream_executor/musa/musa_executor.cc
// 摩尔线程需要实现

StatusOr<std::string> MUSAGpuExecutor::CompileToBinary(
    const llvm::Module& llvm_module) {
  
  // MUSA 有自己的格式，需要软件团队提供转换工具
  std::string ptx = CompileToPtx(llvm_module);
  return ConvertPtxToMusaBinary(ptx);
}
```

---

## 4. 代码生成详细实现 (Code Generation Implementation)

### 4.1 核心组件架构

```
┌─────────────────────────────────────────────────────────┐
│                    XLA Frontend                          │
│               (JAX, TensorFlow, PyTorch)                 │
└────────────────────┬────────────────────────────────────┘
                     │ HLO Graph
┌────────────────────▼────────────────────────────────────┐
│                 MTGPUCompiler                            │
│  ┌────────────────────────────────────────────────┐     │
│  │ HLO Optimization Passes                        │     │
│  │  - Convolution Canonicalization               │     │
│  │  - BFloat16 Normalization                     │     │
│  │  - Layout Assignment                           │     │
│  │  - Algebraic Simplification                    │     │
│  └────────────────┬───────────────────────────────┘     │
└───────────────────┼─────────────────────────────────────┘
                    │ LLVM IR (NVVM-style)
┌───────────────────▼─────────────────────────────────────┐
│              MTGPU LLVM Backend                          │
│  ┌────────────────────────────────────────────────┐     │
│  │ IR Transformations (Critical Step)             │     │
│  │  - NVVM → MUSA intrinsic conversion            │     │
│  │  - BFloat16 operation lowering                 │     │
│  │  - Calling convention updates                  │     │
│  │  - Global variable preservation                │     │
│  └────────────────┬───────────────────────────────┘     │
│                   │ LLVM IR (MUSA-style)                 │
│  ┌────────────────▼───────────────────────────────┐     │
│  │ Compilation Pipeline                           │     │
│  │  1. llvm-link (with libdevice)                 │     │
│  │  2. opt -O2 (optimization)                     │     │
│  │  3. llc (code generation)                      │     │
│  │  4. lld (linking)                              │     │
│  └────────────────┬───────────────────────────────┘     │
└───────────────────┼─────────────────────────────────────┘
                    │ HSACO/MUBIN Binary
┌───────────────────▼─────────────────────────────────────┐
│          MUSA Runtime & Hardware                         │
│            Moore Threads GPU (mp_31)                     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 关键源文件位置

| 组件 | 文件路径 | 说明 |
|------|----------|------|
| **Compiler** | `xla/service/gpu/mtgpu_compiler.{h,cc}` | MTGPU 编译器主类 |
| **LLVM Backend** | `xla/service/gpu/llvm_gpu_backend/mtgpu_backend.{h,cc}` | LLVM IR 转换和编译 |
| **Registration** | `xla/service/gpu/mtgpu_compiler_registration.cc` | 编译器注册 |
| **Platform ID** | `stream_executor/musa/musa_platform_id.h` | MUSA 平台标识 |
| **Intrinsic Rules** | `xla/service/gpu/llvm_gpu_backend/musa_intrinsic.def` | NVVM→MUSA 映射规则 |

### 4.3 编译器初始化

```cpp
// xla/service/gpu/mtgpu_compiler.cc:272-274

MTGPUCompiler::MTGPUCompiler()
    : GpuCompiler(stream_executor::musa::kMUSaPlatformId,
                  mtgpu::TargetTriple(), mtgpu::DataLayout()) {}
```

**编译器注册**（自动完成）：

```cpp
// xla/service/gpu/mtgpu_compiler_registration.cc:22-28

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::musa::kMUSaPlatformId,
      []() { return std::make_unique<xla::gpu::MTGPUCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
```

---

## 5. IR 转换详解 (IR Transformations)

### 5.1 转换流程概览

```
┌─────────────────────────────────────────┐
│ Input: LLVM Module (NVVM intrinsics)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 1. IR Transformations                   │
│    - NVVM → MUSA intrinsics             │
│    - BFloat16 lowering (6 passes)       │
│    - Global var preservation            │
│    - ptx_kernel → mtgpu_kernel          │
└──────────────┬──────────────────────────┘
               │ .ll file
               ▼
┌─────────────────────────────────────────┐
│ 2. llvm-link                            │
│    Link with libdevice.31.ll            │
│    --only-needed flag                   │
└──────────────┬──────────────────────────┘
               │ linked .ll
               ▼
┌─────────────────────────────────────────┐
│ 3. opt -O2                              │
│    Standard LLVM optimizations          │
└──────────────┬──────────────────────────┘
               │ optimized .ll
               ▼
┌─────────────────────────────────────────┐
│ 4. llc                                  │
│    -march=mtgpu                         │
│    -mcpu=mp_31                          │
│    -filetype=obj                        │
└──────────────┬──────────────────────────┘
               │ .o object file
               ▼
┌─────────────────────────────────────────┐
│ 5. lld                                  │
│    -flavor gnu                          │
│    -shared                              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Output: HSACO/MUBIN binary              │
│ (Cached for future use)                 │
└─────────────────────────────────────────┘
```

### 5.2 核心转换函数

**转换函数调用顺序**（`mtgpu_backend.cc:918-1055`）：

```cpp
// Stage 1: IR Transformations (6 passes)
convertNvvmToMusaIntrinsics(*module);        // NVVM → MUSA
convertIRToMusaIntrinsics(*module);          // fpext bfloat → float
convertIRToMusaIntrinsics1(*module);         // fptrunc float → bfloat
convertIRToMusaIntrinsics2(*module);         // fcmp on bfloat
convertIRToMusaIntrinsics3(*module);         // fptosi on bfloat
convertBF16BinOpToMusaIntrinsics(*module);   // fadd/fsub/fmul bfloat
preserveGlobalVars(*module);                 // Prevent optimization
```

### 5.3 NVVM 到 MUSA Intrinsic 转换

**规则定义文件**：`musa_intrinsic.def`

```cpp
struct Rule {
  const char* oldName;  // NVVM intrinsic
  const char* newName;  // MUSA intrinsic
  int type;             // Conversion type
};

const Rule rules[] = {
#include "musa_intrinsic.def"
};
```

**转换类型说明**：

| Type | 说明 | 示例 |
|------|------|------|
| **Type 0** | 直接名称替换 | `llvm.nvvm.read.ptx.sreg.tid.x` → `llvm.musa.read.builtin.local.id.x` |
| **Type 1** | 无参数版本（移除参数） | `llvm.nvvm.barrier0(i32 0)` → `llvm.musa.barrier.sync()` |
| **Type 2** | 整数 Shuffle（直接替换） | `llvm.nvvm.shfl.sync.down.i32` → `llvm.musa.shfl.sync.down.i32` |
| **Type 3** | 浮点 Shuffle（需要 bitcast） | `llvm.nvvm.shfl.sync.down.f32` → `llvm.musa.shfl.sync.down.i32` + bitcast |

**常用 Intrinsic 映射表**：

| NVVM Intrinsic | MUSA Intrinsic | Type |
|----------------|----------------|------|
| `llvm.nvvm.read.ptx.sreg.tid.x` | `llvm.musa.read.builtin.local.id.x` | 0 |
| `llvm.nvvm.read.ptx.sreg.tid.y` | `llvm.musa.read.builtin.local.id.y` | 0 |
| `llvm.nvvm.read.ptx.sreg.tid.z` | `llvm.musa.read.builtin.local.id.z` | 0 |
| `llvm.nvvm.read.ptx.sreg.ctaid.x` | `llvm.musa.read.builtin.group.id.x` | 0 |
| `llvm.nvvm.read.ptx.sreg.ctaid.y` | `llvm.musa.read.builtin.group.id.y` | 0 |
| `llvm.nvvm.read.ptx.sreg.ctaid.z` | `llvm.musa.read.builtin.group.id.z` | 0 |
| `llvm.nvvm.barrier0` | `llvm.musa.barrier.sync` | 1 |
| `llvm.nvvm.shfl.sync.down.i32` | `llvm.musa.shfl.sync.down.i32` | 2 |
| `llvm.nvvm.shfl.sync.down.f32` | `llvm.musa.shfl.sync.down.i32` | 3 |
| `llvm.nvvm.shfl.sync.up.i32` | `llvm.musa.shfl.sync.up.i32` | 2 |
| `llvm.nvvm.shfl.sync.bfly.i32` | `llvm.musa.shfl.sync.bfly.i32` | 2 |
| `llvm.nvvm.shfl.sync.idx.i32` | `llvm.musa.shfl.sync.idx.i32` | 2 |

**Type 3 转换详情**（浮点 Shuffle）：

```cpp
// 原始 IR
call float @llvm.nvvm.shfl.sync.down.f32(i32 %mask, float %val, ...)

// 转换后 IR
%temp_i32 = bitcast float %val to i32
%result_i32 = call i32 @llvm.musa.shfl.sync.down.i32(i32 %temp_i32, ...)
%result = bitcast i32 %result_i32 to float
```

**实现代码**（`mtgpu_backend.cc:841-861`）：

```cpp
case 3: {
  // MUSA only has i32 shuffle, so bitcast float → i32 → shuffle → i32 → float
  llvm::Value *Val = CI->getArgOperand(1);        // float value
  llvm::Value *Offset = CI->getArgOperand(2);
  llvm::Value *Mask = CI->getArgOperand(3);
  llvm::Value *NullPtr = llvm::ConstantPointerNull::get(PtrAS5Ty);

  llvm::Value *TempX = Builder.CreateBitCast(Val, Int32Ty);
  llvm::CallInst *NewCall = llvm::CallInst::Create(
      NewFTy, newF, {TempX, Offset, Mask, NullPtr}, "", CI);
  llvm::Value *Result = Builder.CreateBitCast(NewCall, F32Ty, CI->getName());

  CI->replaceAllUsesWith(Result);
  CI->eraseFromParent();
}
```

### 5.4 BFloat16 支持

**BFloat16 操作转换表**：

| 操作 | LLVM IR | MUSA Intrinsic |
|------|---------|----------------|
| Float → BF16 | `fptrunc float to bfloat` | `llvm.musa.float2bfloat16` |
| BF16 → Float | `fpext bfloat to float` | `llvm.musa.bfloat162float` |
| BF16 加法 | `fadd bfloat` | `llvm.musa.hadd.bf16` |
| BF16 减法 | `fsub bfloat` | `llvm.musa.hsub.bf16` |
| BF16 乘法 | `fmul bfloat` | `llvm.musa.hmul.bf16` |
| BF16 比较 | `fcmp ... bfloat` | 先转 float，再比较 |
| BF16 转整数 | `fptosi bfloat` | 先转 float，再转整数 |

**BF16 → Float 转换**（`mtgpu_backend.cc:677-760`）：

```llvm
; 原始 IR
%result = fpext bfloat %x to float

; 转换后 IR
%result = tail call float @llvm.musa.bfloat162float(bfloat %x)
```

**Intrinsic 声明**：

```cpp
llvm::FunctionType *FTy =
  llvm::FunctionType::get(F32Ty, {BFloatTy}, false);
F32FromBF16 = llvm::Function::Create(FTy,
  llvm::GlobalValue::ExternalLinkage,
  "llvm.musa.bfloat162float", &M);
F32FromBF16->setOnlyReadsMemory();   // readnone
F32FromBF16->setDoesNotThrow();       // nounwind
F32FromBF16->setWillReturn();         // willreturn
```

**Float → BF16 转换**（`mtgpu_backend.cc:718-717`）：

```llvm
; 原始 IR
%result = fptrunc float %x to bfloat

; 转换后 IR
%result = tail call bfloat @llvm.musa.float2bfloat16(float %x)
```

**BF16 比较转换**（`mtgpu_backend.cc:629-675`）：

```llvm
; 原始 IR
%cmp = fcmp une bfloat %a, bfloat %b

; 转换后 IR（MUSA 不支持直接 BF16 比较）
%a_f32 = tail call float @llvm.musa.bfloat162float(bfloat %a)
%b_f32 = tail call float @llvm.musa.bfloat162float(bfloat %b)
%cmp = fcmp une float %a_f32, %b_f32
```

**BF16 转整数**（`mtgpu_backend.cc:585-627`）：

```llvm
; 原始 IR
%i = fptosi bfloat %x to i32

; 转换后 IR
%x_f32 = tail call float @llvm.musa.bfloat162float(bfloat %x)
%i = fptosi float %x_f32 to i32
```

**BF16 二元操作**（`mtgpu_backend.cc:519-583`）：

```llvm
; 原始 IR
%result = fadd bfloat %a, %b
%result = fsub bfloat %a, %b
%result = fmul bfloat %a, %b

; 转换后 IR
%result = tail call bfloat @llvm.musa.hadd.bf16(bfloat %a, bfloat %b)
%result = tail call bfloat @llvm.musa.hsub.bf16(bfloat %a, bfloat %b)
%result = tail call bfloat @llvm.musa.hmul.bf16(bfloat %a, bfloat %b)
```

### 5.5 全局变量保护

**功能**：防止 LLVM 优化掉设备端全局变量

**实现**（`mtgpu_backend.cc:479-518`）：

```cpp
void preserveGlobalVars(llvm::Module &M) {
  // 收集所有全局变量定义
  for (llvm::GlobalVariable &GV : M.globals())
    if (!GV.isDeclaration())
      Keep.insert(&GV);

  // 创建/更新 @llvm.used metadata
  llvm::GlobalVariable *LLVMUsed = M.getGlobalVariable("llvm.used");
  LLVMUsed->setInitializer(ConstantArray::get(ATy, Elements));
}
```

### 5.6 调用约定更新

**ptx_kernel → mtgpu_kernel**：

```cpp
std::regex ptx_kernel_regex("ptx_kernel");
ir_content = std::regex_replace(ir_content, ptx_kernel_regex, "mtgpu_kernel");
```

---

## 6. 编译管道详解 (Compilation Pipeline)

### 6.1 CompileToHsaco 函数

**文件位置**：`xla/service/gpu/llvm_gpu_backend/mtgpu_backend.cc:918-1055`

**完整流程**：

```cpp
absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module,
    const se::GpuComputeCapability& gpu_compute_capability,
    const DebugOptions& debug_options,
    const std::string& compilation_cache_key) {
  
  // Stage 1: Cache Lookup
  std::string gcn_arch_name = "mtgpu";
  uint64_t hash;
  if (HsacoCache::Find(str, hash, gcn_arch_name, hsaco)) {
    VLOG(1) << "HSACO cache hit";
    return hsaco;
  }
  
  // Stage 2: IR Transformations
  convertNvvmToMusaIntrinsics(*module);
  convertIRToMusaIntrinsics(*module);
  convertIRToMusaIntrinsics1(*module);
  convertIRToMusaIntrinsics2(*module);
  convertIRToMusaIntrinsics3(*module);
  convertBF16BinOpToMusaIntrinsics(*module);
  preserveGlobalVars(*module);
  
  // Stage 3: Module Preparation
  // Replace ptx_kernel → mtgpu_kernel
  std::regex ptx_kernel_regex("ptx_kernel");
  ir_content = std::regex_replace(ir_content, ptx_kernel_regex, "mtgpu_kernel");
  
  // Stage 4: llvm-link
  std::string llvmlinkCmd =
    "llvm-link \"" + llFile + "\" \"" + bcFile + "\" --only-needed -S -o \"" + linkFile + "\"";
  
  // Stage 5: opt -O2
  std::string optCmd = "opt -O2 \"" + linkFile + "\" -S -o \"" + optFile + "\"";
  
  // Stage 6: llc (code generation)
  std::string llcCmd = "llc \"" + optFile + "\" -march=mtgpu -mcpu=mp_31 "
                       "-filetype=obj -o \"" + objFile + "\"";
  
  // Stage 7: lld (linking)
  std::string lldCmd = "lld -flavor gnu -shared \"" + objFile + "\" -o \"" + mubinFile + "\"";
  
  // Stage 8: Binary Read & Cache
  std::ifstream bin(mubinFile, std::ios::binary | std::ios::ate);
  hsaco.resize(sz);
  bin.read(reinterpret_cast<char*>(hsaco.data()), sz);
  
  HsacoCache::Add(str, hash, gcn_arch_name, hsaco);
  return hsaco;
}
```

### 6.2 工具链要求

**必需的外部工具**：

| 工具 | 用途 | 来源 |
|------|------|------|
| `llvm-link` | Module 链接器 | LLVM |
| `opt` | LLVM 优化器 | LLVM |
| `llc` | 代码生成器（需 MTGPU backend） | 摩尔线程 LLVM |
| `lld` | 链接器（GNU flavor） | LLVM |

**环境变量**：

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `LLVM_PATH` | LLVM 工具路径 | `$MUSA_ROOT/llvm/bin` |
| `TF_MUSA_KEEP_XLA_TEMPFILES` | 保留临时文件 | `false` |

### 6.3 设备库链接

**MUSDL (MUSA Device Libraries)**：

```cpp
// xla/service/gpu/llvm_gpu_backend/mtgpu_backend.cc:112-136

std::vector<std::string> GetMUSDLPaths(
    const std::string& gcn_arch_name,
    const std::string& musdl_dir_path) {
  std::vector<std::string> musdl_filenames = {
    "libdevice.31.bc",      // Version 31 device library
    "libdevice.bc",         // Generic device library
    "libdevice.mthg.bc"     // Moore Threads hardware-specific
  };
  
  // Architecture-specific bitcode
  result.push_back(tsl::io::JoinPath(
      musdl_dir_path,
      absl::StrCat("oclc_isa_version_", mtgpu_version, ".bc")));
}
```

**链接条件**：仅当模块包含设备函数调用时才链接

```cpp
absl::Status LinkMUSDLIfNecessary(llvm::Module* module,
                                  std::string gcn_arch_name,
                                  const std::string& musdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();  // Skip if no device functions used
  }
  return LinkWithBitcodeVector(module,
                               GetMUSDLPaths(gcn_arch_name, musdl_dir_path));
}
```

---

## 7. 缓存机制 (Caching Mechanism)

### 7.1 HSACO 缓存

**文件位置**：`mtgpu_backend.cc:138-191`

```cpp
struct HsacoCacheEntry {
  uint64_t hash;              // Hash of IR string
  std::string ir;             // Full IR text
  std::string gfx;            // Architecture name
  std::vector<uint8_t> hsaco; // Compiled binary
};

struct HsacoCache {
 protected:
  std::vector<HsacoCacheEntry> cache;
  std::mutex m_mutex;
  int request_count = 0;
  int hit_count = 0;

 public:
  static bool Find(...);
  static void Add(...);
};
```

### 7.2 缓存查找

```cpp
bool HsacoCache::Find(const std::string& ir, uint64_t& hash,
                      const std::string& gfx, std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  hash = std::hash<std::string>{}(ir);

  for (auto& x : g_hsacoCache.cache) {
    if (x.hash != hash) continue;
    if (x.gfx != gfx) continue;
    if (x.ir != ir) continue;  // Full IR comparison for collision handling
    hsaco = x.hsaco;
    return true;
  }
  return false;
}
```

### 7.3 缓存统计

```cpp
g_hsacoCache.request_count++;
if (hit) g_hsacoCache.hit_count++;
if (!(g_hsacoCache.request_count % 50))
  VLOG(1) << "HSACO cache: " << g_hsacoCache.request_count << " requests, "
          << g_hsacoCache.hit_count << " hits";
```

**典型命中率**：40-60%（迭代工作负载）

---

## 8. BFloat16 配置 (BFloat16 Configuration)

### 8.1 卷积 BFloat16 支持

**文件位置**：`mtgpu_compiler.cc:82-103`

```cpp
class ConvBfloat16Support : public FloatSupport {
 public:
  explicit ConvBfloat16Support()
      : FloatSupport(BF16),
        is_conv_bf16_supported_(true) {}  // ENABLED
};
```

### 8.2 Matmul BFloat16 支持

**文件位置**：`mtgpu_compiler.cc:105-126`

```cpp
class MatmulBfloat16Support : public FloatSupport {
 public:
  explicit MatmulBfloat16Support()
      : FloatSupport(BF16),
        is_matmul_bf16_supported_(false) {}  // DISABLED
};
```

**说明**：卷积可使用 BF16，但 Matmul BF16 需要额外调优或库支持（当前禁用）

### 8.3 HLO 优化管道中的 BFloat16

```cpp
HloPassPipeline pipeline("conv_canonicalization");

// BFloat16 support for convolutions (enabled)
ConvBfloat16Support conv_bf16_support;
pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

// Inlining and simplification
pipeline.AddPass<CallInliner>();
pipeline.AddPass<TupleSimplifier>();

// Algebraic simplification to fixed point
pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(algsimp_options, gpu_version);

// Final cleanup
pipeline.AddPass<ConvertMover>();
pipeline.AddPass<HloConstantFolding>();
```

---

## 9. Kernel 实现与执行 (Kernel Implementation)

### 9.1 Kernel 来源

| 来源                 | 生成方式                   | 摩尔线程工作               |
| -------------------- | -------------------------- | -------------------------- |
| **XLA 自动生成**     | HLO → LLVM IR → PTX/二进制 | 确保 LLVM 能生成 MUSA 代码 |
| **手写 CUDA Kernel** | 开发者手写                 | 使用 MUSIFY 转换或重写     |
| **自定义算子**       | FFI / Custom Call          | 实现 MUSA 版本             |

### 9.2 运行时执行流程

```cpp
// xla/stream_executor/musa/musa_executor.cc
// 摩尔线程需要实现

// 1. Kernel 加载
Status MUSAGpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                   KernelBase* kernel) {
  // 从编译产物中加载 Kernel
  musaModule_t module;
  MUSA_RETURN_IF_ERROR(musaModuleLoadData(&module, spec.binary_data()));
  
  musaFunction_t function;
  MUSA_RETURN_IF_ERROR(musaModuleGetFunction(&function, module, spec.name()));
  
  kernel->set_gpu_function(function);
  return OkStatus();
}

// 2. Kernel 启动
Status MUSAGpuExecutor::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const KernelBase& kernel,
                                const KernelArgsArrayBase& args) {
  void** params = const_cast<void**>(args.argument_addresses().data());
  
  MUSA_RETURN_IF_ERROR(musaLaunchKernel(
      kernel.gpu_function(),
      block_dims.x, block_dims.y, block_dims.z,
      thread_dims.x, thread_dims.y, thread_dims.z,
      args.shared_memory_bytes(),
      musa_stream_,
      params,
      nullptr));
  
  return OkStatus();
}
```

### 9.3 内存管理

```cpp
// 设备内存分配
void* MUSAGpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  void* ptr = nullptr;
  if (memory_space == 0) {
    musaMalloc(&ptr, size);  // 设备内存
  } else {
    musaMallocManaged(&ptr, size, musaMemAttachGlobal);  // 统一内存
  }
  return ptr;
}

// 内存拷贝
Status MUSAGpuExecutor::Memcpy(void* dst, const void* src, uint64_t size) {
  return ToStatus(musaMemcpy(dst, src, size, musaMemcpyDefault));
}

Status MUSAGpuExecutor::MemcpyAsync(void* dst, const void* src, 
                                     uint64_t size, Stream* stream) {
  auto* musa_stream = static_cast<MusaStream*>(stream);
  return ToStatus(musaMemcpyAsync(dst, src, size, 
                                   musaMemcpyDefault, 
                                   musa_stream->musa_stream()));
}
```

---

## 10. 开发指南 (Development Guide)

### 10.1 添加新的 Intrinsic 映射

**步骤 1**：在 `musa_intrinsic.def` 中添加规则

```cpp
{"llvm.nvvm.your.intrinsic", "llvm.musa.your.intrinsic", 0},
```

**步骤 2**：选择转换类型

| Type | 适用场景 |
|------|----------|
| 0 | 直接名称替换 |
| 1 | 无参数版本（如 barrier） |
| 2 | 整数 shuffle |
| 3 | 浮点 shuffle（需要 bitcast） |

**步骤 3**：重新编译

```bash
bazel build --config=musa //xla/service/gpu:mtgpu_compiler
```

### 10.2 启用 BFloat16 Matmul

**文件**：`mtgpu_compiler.cc:109`

```cpp
explicit MatmulBfloat16Support()
    : FloatSupport(BF16),
      is_matmul_bf16_supported_(true) {}  // 改为 true
```

**测试要求**：
- 验证 muBlas BF16 matmul 支持
- 验证数值精度
- 性能基准测试

### 10.3 调试编译问题

**启用详细日志**：

```bash
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE=mtgpu_backend=3,mtgpu_compiler=3
export TF_MUSA_KEEP_XLA_TEMPFILES=1
```

**检查中间文件**：

```bash
ls /tmp/temp_*
# temp_xxx.ll          - 原始 IR
# temp_xxx_link.ll     - 链接 libdevice 后
# temp_xxx_opt.ll      - 优化后
# temp_xxx.o           - 目标文件
# temp_xxx.mubin       - 最终二进制
```

**验证转换**：

```bash
diff temp_xxx.ll temp_xxx_link.ll
# 应显示 ptx_kernel → mtgpu_kernel
# 应显示 NVVM → MUSA intrinsics
```

---

## 11. 故障排查 (Troubleshooting)

### 11.1 编译失败："llvm-link not found"

**原因**：LLVM 工具不在 PATH 中

**解决方案**：

```bash
export LLVM_PATH=/usr/local/musa/llvm
export PATH=$LLVM_PATH/bin:$PATH
```

或设置 `MUSA_HOME`：

```bash
export MUSA_HOME=/usr/local/musa
```

### 11.2 编译失败："cannot open libdevice.31.ll"

**原因**：硬编码路径不存在

**解决方案**：创建符号链接或修改代码：

```cpp
// 修改 mtgpu_backend.cc:999
std::string bcFile = "/path/to/your/libdevice.31.ll";
```

### 11.3 运行时 Kernel 加载失败

**检查清单**：
1. HSACO/MUBIN 格式是否正确
2. 架构版本是否匹配（mp_31）
3. MUSA Driver 版本是否兼容

### 11.4 性能问题

**可能原因**：
- Autotuning 未实现（使用默认算法）
- BFloat16 Matmul 被禁用（使用 FP32）
- 缓存未命中

**诊断**：

```bash
# 查看缓存命中率
export TF_CPP_VMODULE=mtgpu_backend=2

# 检查是否使用最优算法
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
```

---

## 12. 最小集成 Demo (Minimal Integration Demo)

### 12.1 环境配置

```bash
# 1. 安装 MUSA SDK
export MUSA_PATH=/usr/local/musa
export PATH=$MUSA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_PATH/lib:$LD_LIBRARY_PATH

# 2. 设置摩尔线程 PJRT 插件路径
export PJRT_NAMES_AND_LIBRARY_PATHS="musa:/path/to/libmusa_pjrt_plugin.so"

# 3. 可选：调试标志
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export MUSA_VISIBLE_DEVICES=0
```

### 12.2 Python 示例代码

```python
"""
摩尔线程 OpenXLA 集成测试示例
展示从 TF 模型到 XLA 编译再到 MUSA GPU 执行的完整流程
"""

import os
import tensorflow as tf

# ========== 摩尔线程特定配置 ==========
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = \
    'musa:/usr/local/musa/lib/libmusa_pjrt_plugin.so'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'
os.environ['MUSA_VISIBLE_DEVICES'] = '0'
# =====================================

print(f"TensorFlow 版本: {tf.__version__}")
print(f"可用设备: {tf.config.list_physical_devices()}")

# XLA 编译测试
@tf.function(jit_compile=True)
def compute(x, y):
    """会被 XLA 编译并执行在 MUSA GPU 上"""
    return tf.nn.relu(tf.matmul(x, y) + tf.reduce_sum(y))

# 创建测试数据
x = tf.random.normal([128, 256], dtype=tf.float32)
y = tf.random.normal([256, 512], dtype=tf.float32)

# 预热（触发 XLA 编译）
_ = compute(x, y)

# 正式执行
result = compute(x, y)
print(f"输出形状: {result.shape}")
print(f"设备放置: {result.device}")
```

### 12.3 PJRT C++ 插件框架

```cpp
// xla/pjrt/musa/musa_pjrt_plugin.cc
// 摩尔线程需要实现

#include "xla/pjrt/c/pjrt_c_api.h"
#include "musa_runtime_api.h"

// 导出函数 - PJRT 插件入口
extern "C" PJRT_EXPORT const PJRT_Api* GetPjrtApi(void) {
  static const PJRT_Api api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .priv = nullptr,
    
    // 客户端管理
    .Client_Create = MusaClient_Create,
    .Client_Destroy = MusaClient_Destroy,
    .Client_Devices = MusaClient_Devices,
    
    // 设备管理
    .Device_Id = MusaDevice_Id,
    .Device_Kind = MusaDevice_Kind,
    
    // 内存管理
    .Buffer_Create = MusaBuffer_Create,
    .Buffer_ToHostBuffer = MusaBuffer_ToHostBuffer,
    
    // 执行（加载已编译的程序并执行）
    .LoadedExecutable_Create = MusaExecutable_Create,
    .LoadedExecutable_Execute = MusaExecutable_Execute,
    
    // ... 其他接口
  };
  return &api;
}
```

---

## 13. 当前限制与未来工作 (Current Limitations)

### 13.1 Autotuning 未实现

**被 Stub 的函数**：
- `AddConvAndGemmAutotuningPasses()` - 立即返回
- `AddGemmFusionAutotuningPasses()` - 立即返回

**影响**：无法选择最优 GEMM/卷积算法，仅使用默认值

**代码**：

```cpp
absl::Status MTGPUCompiler::AddConvAndGemmAutotuningPasses(...) {
  // TODO: Implement autotuning
  return absl::OkStatus();
}
```

### 13.2 禁用的 HLO 优化

**注释掉的 Pass**：

```cpp
// pipeline.AddPass<GpusolverRewriter>(...);
// pipeline.AddPass<ConvRewriter>(gpu_version);
// pipeline.AddPass<ConvPaddingLegalization>();
// pipeline.AddPass<CudnnFusedConvRewriter>(...);
// pipeline.AddPass<DotDimensionMerger>();
// pipeline.AddPass<CublasPadForGemms>(...);
// pipeline.AddPass<TriangularSolveRewriter>();
```

**原因**：可能等待验证或 MUSA 库支持

### 13.3 硬编码路径

```cpp
std::string bcFile = "/data/moon/github/openxla/third_party/gpus/musa/libdevice.31.ll";
```

**位置**：`mtgpu_backend.cc:999`

**问题**：非可移植路径，应使用环境变量或构建配置

### 13.4 外部工具依赖

**使用 `std::system()` 调用**：
- `llvm-link`
- `opt`
- `llc`
- `lld`

**问题**：
- Shell 注入风险
- 平台依赖
- 错误处理受限
- 无法捕获详细诊断信息

**建议**：使用 LLVM 库 API 替代外部工具

---

## 附录 A: 参考资源

- [OpenXLA XLA](https://github.com/openxla/xla)
- [Intel Extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla)
- [PJRT Plugin RFC](https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md)
- [摩尔线程开发者网站](https://developer.mthreads.com/)
