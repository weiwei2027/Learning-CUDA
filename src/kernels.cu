#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// ============================================================================
// 第一题：trace 函数实现
// ============================================================================

/**
 * @brief CUDA kernel for computing matrix trace using parallel reduction
 *
 * 此内核函数使用共享内存和并行归约算法高效计算矩阵的迹（对角线元素之和）。
 * 算法特点：
 * 1. 每个线程负责加载一个对角线元素到共享内存
 * 2. 使用树形归约在共享内存中进行部分和计算
 * 3. 最终每个block产生一个部分和结果
 *
 * @tparam T 数据类型模板参数
 * @param d_input 输入矩阵的设备指针（行主序存储）
 * @param d_output 输出部分和的设备指针
 * @param n 对角线元素数量（min(rows, cols)）
 * @param cols 矩阵列数
 */
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_output, size_t n, size_t cols) {
    // 分配共享内存用于线程间通信和归约操作
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    // 计算线程ID和全局索引
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载对应的对角线元素，超出范围则填充0
    sdata[tid] = (i < n) ? d_input[i * cols + i] : T(0);
    __syncthreads(); // 确保所有线程完成数据加载

    // 树形归约算法：逐步将相邻元素相加
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // 归约步骤
        }
        __syncthreads(); // 同步确保内存一致性
    }

    // 每个block的第一个线程将部分和写入全局内存
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // TODO: Implement the trace function
    // 确定对角线元素数量（处理非方阵情况）
    size_t n = (rows < cols) ? rows : cols;

    // 边界条件检查
    if (n == 0) {
        return T(0);
    }

    // 配置CUDA执行参数
    const int blockSize = 256;  // 每个block的线程数
    const int gridSize = (n + blockSize - 1) / blockSize;  // block数量

    // 分配GPU内存
    T* d_input = nullptr;
    T* d_output = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(T)));      // 输入矩阵
    RUNTIME_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));        // 部分和结果

    // 数据传输：主机到设备
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice));

    // 启动CUDA内核执行并行迹计算
    traceKernel<<<gridSize, blockSize, blockSize * sizeof(T)>>>(d_input, d_output, n, cols);
    RUNTIME_CHECK(cudaGetLastError());  // 错误检查

    // 收集各block的部分和结果
    std::vector<T> h_partial(gridSize);
    RUNTIME_CHECK(cudaMemcpy(h_partial.data(), d_output, gridSize * sizeof(T), cudaMemcpyDeviceToHost));

    // CPU端最终归约求和
    T result = T(0);
    for (int i = 0; i < gridSize; i++) {
        result += h_partial[i];
    }

    // 清理GPU资源
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return result;
}


// ============================================================================
// 第二题：Flash Attention 函数实现
// ============================================================================

/**
 * @brief Flash Attention CUDA核函数 - 标准两遍Softmax算法
 *
 * 使用标准两遍softmax算法，与PyTorch的scaled_dot_product_attention行为完全一致：
 * 第一遍：计算所有scores，找到max_score，计算sum_exp
 * 第二遍：使用max_score和sum_exp计算softmax权重和输出
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal
) {
    // 计算全局线程ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算总线程数并进行边界检查
    int total_threads = batch_size * tgt_seq_len * query_heads;
    if (tid >= total_threads) return;

    // 从全局线程ID分解出batch、位置和head索引
    int batch_idx = tid / (tgt_seq_len * query_heads);  // 当前处理的batch索引
    int rem = tid % (tgt_seq_len * query_heads);        // 剩余索引用于计算位置和head
    int tgt_pos = rem / query_heads;                    // 目标序列位置
    int head_idx = rem % query_heads;                   // 查询头索引

    // 计算KV头索引（支持Grouped Query Attention）
    int kv_head_idx = head_idx / (query_heads / kv_heads);

    // 计算内存布局的步长
    int q_stride = query_heads * head_dim;     // Query张量中每个位置的步长
    int kv_stride = kv_heads * head_dim;       // KV张量中每个位置的步长
    int batch_q_offset = batch_idx * tgt_seq_len * q_stride;    // 当前batch在Query中的偏移
    int batch_kv_offset = batch_idx * src_seq_len * kv_stride;  // 当前batch在KV中的偏移

    // 获取当前线程处理的Query向量指针
    const T* q_ptr = q + batch_q_offset + tgt_pos * q_stride + head_idx * head_dim;

    // 将Query向量加载到寄存器中以提高访问速度
    float q_local[128];
    for (int d = 0; d < head_dim; d++) {
        q_local[d] = static_cast<float>(q_ptr[d]);
    }

    // 计算缩放因子，用于点积注意力的数值稳定性
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // 确定source序列的结束位置（因果掩码处理）
    int src_end = is_causal ? (tgt_pos + 1) : src_seq_len;

    // ========== 第一遍：计算所有attention scores并找到最大值 ==========
    float max_score = -INFINITY;

    for (int src_pos = 0; src_pos < src_end; src_pos++) {
        // 获取当前source位置的Key向量指针
        const T* k_ptr = k + batch_kv_offset + src_pos * kv_stride + kv_head_idx * head_dim;

        // 计算Query和Key的点积（使用float提高精度）
        float score = 0.0f;
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            score += q_local[d] * static_cast<float>(k_ptr[d]);
            score += q_local[d+1] * static_cast<float>(k_ptr[d+1]);
            score += q_local[d+2] * static_cast<float>(k_ptr[d+2]);
            score += q_local[d+3] * static_cast<float>(k_ptr[d+3]);
        }
        for (; d < head_dim; d++) {
            score += q_local[d] * static_cast<float>(k_ptr[d]);
        }
        // 应用缩放因子
        score *= scale;

        // 存储score并更新最大值
        if (score > max_score) {
            max_score = score;
        }
    }

    // ========== 第二遍：计算softmax权重和最终输出 ==========
    // 初始化softmax分母和输出累加器
    float sum_exp = 0.0f;           // softmax分母：所有exp(score)的和
    float out[128];                 // 输出累加器数组

    // 初始化输出数组为零
    for (int d = 0; d < head_dim; d++) {
        out[d] = 0.0f;
    }

    // 遍历所有source位置计算softmax权重并累加到输出
    for (int src_pos = 0; src_pos < src_end; src_pos++) {
        // 获取当前source位置的Key和Value向量指针
        const T* k_ptr = k + batch_kv_offset + src_pos * kv_stride + kv_head_idx * head_dim;
        const T* v_ptr = v + batch_kv_offset + src_pos * kv_stride + kv_head_idx * head_dim;

        // 计算当前Query-Key对的attention score
        float score = 0.0f;
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            score += q_local[d] * static_cast<float>(k_ptr[d]);
            score += q_local[d+1] * static_cast<float>(k_ptr[d+1]);
            score += q_local[d+2] * static_cast<float>(k_ptr[d+2]);
            score += q_local[d+3] * static_cast<float>(k_ptr[d+3]);
        }
        for (; d < head_dim; d++) {
            score += q_local[d] * static_cast<float>(k_ptr[d]);
        }
        score *= scale;  // 应用缩放因子

        // 计算softmax权重：exp(score - max_score)以保证数值稳定性
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;  // 累加到分母

        // 使用softmax权重加权Value向量并累加到输出
        for (int d = 0; d < head_dim; d++) {
            out[d] += exp_score * static_cast<float>(v_ptr[d]);
        }
    }

    // 执行最终的softmax归一化：除以分母得到最终输出
    int d = 0;
    for (; d + 3 < head_dim; d += 4) {
        out[d]     /= sum_exp;
        out[d + 1] /= sum_exp;
        out[d + 2] /= sum_exp;
        out[d + 3] /= sum_exp;
    }
    for (; d < head_dim; d++) {
        out[d] /= sum_exp;
    }

    // 写回最终结果
    T* o_ptr = o + batch_q_offset + tgt_pos * q_stride + head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        o_ptr[d] = static_cast<T>(out[d]);
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // TODO: Implement the flash attention function
    // 计算各个张量在设备上的内存大小
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

    // 声明设备端指针并分配GPU内存
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size));     // 为Query分配GPU内存
    RUNTIME_CHECK(cudaMalloc(&d_k, kv_size));    // 为Key分配GPU内存
    RUNTIME_CHECK(cudaMalloc(&d_v, kv_size));    // 为Value分配GPU内存
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size));     // 为Output分配GPU内存

    // 将输入数据从主机内存复制到设备内存
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));  // 复制Query数据
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice)); // 复制Key数据
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice)); // 复制Value数据

    // 配置CUDA kernel执行参数
    int total_threads = batch_size * target_seq_len * query_heads;  // 总线程数
    int blockSize = 256;                                         // 每个block的线程数
    int gridSize = (total_threads + blockSize - 1) / blockSize;  // 计算所需的block数量

    // 启动Flash Attention CUDA kernel
    flashAttentionKernel<<<gridSize, blockSize>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal
    );
    RUNTIME_CHECK(cudaGetLastError());  // 检查kernel启动是否出错

    // 将计算结果从设备内存复制回主机内存
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

    // 释放GPU内存资源
    RUNTIME_CHECK(cudaFree(d_q));   // 释放Query内存
    RUNTIME_CHECK(cudaFree(d_k));   // 释放Key内存
    RUNTIME_CHECK(cudaFree(d_v));   // 释放Value内存
    RUNTIME_CHECK(cudaFree(d_o));   // 释放Output内存
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
