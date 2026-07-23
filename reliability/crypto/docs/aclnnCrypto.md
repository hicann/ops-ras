# aclnnCrypto

## 产品支持情况


| 产品                                                    | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>   |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |      ×     |

## 功能说明

- 接口功能：分为读秘钥功能和加解密功能。读秘钥功能获取spdm派生的加解密秘钥并写入device侧tensor，加解密功能使用已读取的秘钥对device侧tensor做加密/解密。

  读秘钥：读密钥前，host 和 device 已经通过 spdm 派生了加解密秘钥，且调用者已经获取到要读取的秘钥的keyId。读秘钥时，通过 uds 从 KMS-proxy 按key_id, alg_type, key_type 获取秘钥, 并写入device侧的 aclTensor key 中。

  加解密：加解密时，采用标准 AES_CTR_128 和 AES_GCM_128 算法计算加解密。使用的秘钥为传入的 device侧 aclTensor key (已经由 读秘钥功能 写入)。使用的 iv 和 tag 可由 host 传到 device 侧。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnCryptoGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCrypto”接口执行计算。
```c++
aclnnStatus aclnnCryptoAicpuGetWorkspaceSize(
  const aclTensor      *key,
  const aclTensor      *inputText,
  aclTensor            *outputText,
  const aclTensor      *iv,
  const aclTensor      *opConfig,
  aclTensor            *tag,
  aclTensor            *aad,
  aclTensor            *out,
  uint64_t             *workspaceSize,
  aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnCryptoAicpu(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCryptoGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 361px">
    <col style="width: 213px">
    <col style="width: 110px">
    <col style="width: 110px">
    <col style="width: 110px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>key(aclTensor*)</td>
        <td>输入</td>
        <td>算子使用的秘钥。</td>
        <td>不支持空Tensor。</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputText (aclTensor*) </td>
        <td>输入</td>
        <td>算子加解密输入的明文或密文。</td>
        <td>
            支持空Tensor，读密钥时可为空。
            形状和类型需要与outputText一致。</td>
        <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BFLOAT16</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputText(aclTensor*)</td>
        <td>输入/输出</td>
        <td>算子加解密输出的明文或密文。</td>
        <td>支持空Tensor，读密钥时可为空。形状和类型需要与inputText一致。</td>
        <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BFLOAT16</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>iv (aclTensor*)</td>
        <td>输入</td>
        <td>算子加解密使用的IV。</td>
        <td>支持空Tensor，读密钥时可为空。</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>opConfig (aclTensor*)</td>
        <td>输入</td>
        <td>算子参数设置，结构为 {version, mode, alg_type, key_type, key_id, device_id}。</td>
        <td>不支持空Tenosr。version 表示算子版本，取值范围为[1,2], 1 表示支持A2/A3的版本， 2 表示支持A5的版本；mode 表示算子工作模式，取值范围为[0,2], 0 表示读取秘钥，1 表示加密模式，2 表示解密模式；alg_type 表示算法类型，取值范围为[1,2], 1 表示 AES_CTR_128 算法，2 表示 AES_GCM_128 算法；key_type 表示秘钥类型，取值范围为[1], 默认为 1，表示秘钥保存在 key 指向的张量内；key_id 表示秘钥id，取值范围为[0,UINT32_MAX], 从 0 开始，key_id 越大说明秘钥越新；device_id 表示设备id, 取值范围为[0,63], 从 0 开始</td>
        <td>UINT32</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>tag (aclTensor*)</td>
        <td>输入/输出</td>
        <td>加解密使用的tag。</td>
        <td>支持空Tensor, 读密钥和aes-ctr-128加解密时可为空Tensor。</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>√</td>
      </tr>
      <tr>
        <td>aad (aclTensor*)</td>
        <td>输入/输出</td>
        <td>算子使用的辅助信息。</td>
        <td>仅支持空Tensor，目前不支持传入非空Tensor。</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>任意维度</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>输出</td>
        <td>状态码，标识算子执行成功还是失败。</td>
        <td>不支持空Tensor。</td>
        <td>UINT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)*</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  aclnnStatus:返回状态码，具体参见 [aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)

  第一段接口完成入参校验，出现以下场景时报错:

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的指针存在空指针</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>算子输入的数据类型和数据格式不在支持的范围之内</td>
    </tr>
  </tbody>
  </table>

## aclnnCrypto

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段aclnnCryptoGetWorkspaceSize接口获取</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>返回op执行器，包含了算子计算流程</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream</td>
    </tr>
  </tbody>
  </table>

- **返回值**

    返回aclnnStatus状态码，具体见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)

## 约束说明

- 确定性计算： aclnnCrypto 默认确定性实现。

- 入参约束：
  1. opConfig.mode = 0 (读取秘钥) 时，inputText, outputText, iv, tag 可以为空(不为空但是类型和形状符合约束也不会报错)
  2. key, opConfig, out 不能为空
  3. inputText 和 outputText 不为空时，形状和大小要相同(类型相同，维数相同，每一维的大小相同)
  4. 不为空的入参类型必须符合表格里的类型约束
  5. opConfig.algType = 1 (AES-CTR-128算法) 时，tag 可以为空(不为空也不会使用该参数)
  6. opConfig.algType = 2 (AES-GCM-128算法) 时，tag 不能为空
  7. opConfig 的参数必须符合定义，version 取值 1,2，mode 取值 0,1,2, algType 取值 1,2, keyType 取值 1，keyId 取值 [0, UINT32_MAX], deviceId 取值 [0,63]
  8. 读秘钥时必须传入正确的 keyId (一个 keyId 对应一个spdm 已派生的加解密秘钥)，否则会报错

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "openssl/evp.h"
#include "openssl/rand.h"
#include "aclnnop/aclnn_crypto.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

enum CryptoAlgType {
    AES_CTR_128 = 1,
    AES_GCM_128 = 2
};

enum CryptoMode {
    CRYPTO_ENC = 1,
    CRYPTO_DEC = 2
};

enum CryptoVersion {
    CRYPTO_VERSION_A2_A3 = 1,
    CRYPTO_VERSION_A5 = 2
};

enum CryptoKeyType {
    CRYPTO_KEY_RAW = 1,       // key
    CRYPTO_KEY_HANDLE = 2,    // key / key handle
    CRYPTO_KEY_CONTEXT = 3    // key
};

struct OpConfig {
    uint32_t version; //A2A3version == 1 A5version == 2
    uint32_t mode;  // 1 enc, 2 dec
    uint32_t alg_type; // CryptoAlgType 1 aes-128-ctr, 2 aes-128-gcm
    uint32_t key_type; // 1
    uint32_t key_id; // 1
    uint32_t device_id; // 1
};

int aes_128_ctr_encrypt(const unsigned char *key, const unsigned char *iv,
                        const unsigned char *plaintext, int plaintext_len,
                        unsigned char *ciphertext) {
    EVP_CIPHER_CTX *ctx = NULL;
    int len = 0, ciphertext_len = 0;

    // 1. 创建并初始化加密上下文
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;

    // 2. 初始化加密算法：AES-128-CTR
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }

    // 3. 分块加密（支持任意长度数据）
    if (1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;

    // 4. 处理最后一块（CTR模式下此步骤无实际数据，但必须调用）
    if (1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;

    // 5. 释放上下文
    EVP_CIPHER_CTX_free(ctx);
    if (ciphertext_len != plaintext_len) {
        return -1;
    }
    return 0;
};

int aes_128_ctr_decrypt(const unsigned char *key, unsigned char *iv,
                        const unsigned char *ciphertext, int ciphertext_len,
                        unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = NULL;
    int len = 0, plaintext_len = 0;

    // 1. 创建并初始化解密上下文
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;

    // 2. 初始化解密算法：AES-128-CTR
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }

    // 3. 分块解密
    if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;

    // 4. 处理最后一块（CTR模式下无实际数据，但必须调用）
    if (1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;
    // 5. 释放上下文
    EVP_CIPHER_CTX_free(ctx);

    if (ciphertext_len != plaintext_len) {
        return -1;
    }
    return 0;
};

#define GCM_TAG_SIZE 16

int aes_128_gcm_encrypt(unsigned char *key, unsigned char *iv,
                        const unsigned char *plaintext, int plaintext_len,
                        unsigned char *ciphertext, unsigned char *tag) {

    EVP_CIPHER_CTX *ctx = nullptr;
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, reinterpret_cast<unsigned char*>(key), iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    int len = 0, ciphertext_len = 0;

    if (1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;

    if (1 != EVP_EncryptFinal_ex(ctx, nullptr, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_SIZE, tag);
    EVP_CIPHER_CTX_free(ctx);
    if (ciphertext_len != plaintext_len) {
        return -1;
    }
    return 0;
};

int aes_128_gcm_decrypt(unsigned char *key, unsigned char *iv,
                        const unsigned char *ciphertext, int ciphertext_len,
                        unsigned char *plaintext, unsigned char* tag) {
    EVP_CIPHER_CTX *ctx = nullptr;
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, reinterpret_cast<unsigned char*>(key), iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    int len = 0, plaintext_len = 0;
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, GCM_TAG_SIZE, (void *)tag);
    if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -2;
    }
    plaintext_len += len;

    if (1 != EVP_DecryptFinal_ex(ctx, nullptr, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -3;
    }
    // 释放
    EVP_CIPHER_CTX_free(ctx);
    // GCM 密文长度 = 明文长度
    if (plaintext_len != ciphertext_len) {
        return -4;
    }
    return 0;
};

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int ExpectEQ(unsigned char *a, unsigned char *b, int msg_len)
{
    for (int i = 0; i < msg_len; ++i) {
        if (a[i]!=b[i]) {
            return -1;
        }
    }
    return 0;
}

void HostCrypto(unsigned char *key, unsigned char *iv, int text_len, unsigned char *input_data, unsigned char *out_data, uint32_t mode = 1)
{
    int res = 0;
    if (mode == 1) {
        res = aes_128_ctr_encrypt(reinterpret_cast<unsigned char *>(key), reinterpret_cast<unsigned char *>(iv), input_data, text_len, reinterpret_cast<unsigned char *>(out_data));
    }
    else if (mode == 2) {
        res = aes_128_ctr_decrypt(reinterpret_cast<unsigned char *>(key), reinterpret_cast<unsigned char *>(iv), input_data, text_len, reinterpret_cast<unsigned char *>(out_data));
    }
    if (res != 0) {
        printf("host crypto failure: %d\n", res);
    }
    std::cout<<"crypto res: "<<res<<"\n";
    for (int i = 0; i < text_len; ++i) {
        if (i < 10) {
            LOG_PRINT("output[%d] = %d\n", i, out_data[i]);
        }
    }
    std::cout<<"\n";
}

void HostCryptoGcm(unsigned char *key, unsigned char *iv, int text_len, unsigned char *input_data, unsigned char *out_data, unsigned char* tag, uint32_t mode = 1)
{
    int res = 0;
    if (mode == 1) {
        res = aes_128_gcm_encrypt(reinterpret_cast<unsigned char *>(key), reinterpret_cast<unsigned char *>(iv), input_data, text_len, reinterpret_cast<unsigned char *>(out_data), tag);
    }
    else if (mode == 2) { // 先加密再解密
        std::vector<uint8_t> org_data(text_len, 1);
        res = aes_128_gcm_encrypt(reinterpret_cast<unsigned char *>(key), reinterpret_cast<unsigned char *>(iv), reinterpret_cast<unsigned char *>(org_data.data()), text_len, reinterpret_cast<unsigned char *>(input_data), tag);
        if (res != 0) {
            printf("host crypto gcm enc failure: %d\n", res);
        }
        for (int i = 0; i < 16; ++i) printf("%d ", (uint32_t)(tag[i]));
        printf("\n");

        res = aes_128_gcm_decrypt(reinterpret_cast<unsigned char *>(key), reinterpret_cast<unsigned char *>(iv), input_data, text_len, reinterpret_cast<unsigned char *>(out_data), tag);
    }
    if (res != 0) {
        printf("host crypto failure: %d\n", res);
    }
    std::cout<<"gcm crypto res: ====================="<<res<<"\n";
    for (int i = 0; i < text_len; ++i) {
        if (i < 10) {
            LOG_PRINT("output[%d] = %d\n", i, (int)out_data[i]);
        }
    }
    //
    std::cout<<"host gcm tag is: ===================\n";
    for (int i = 0; i < 16; ++i) {
        LOG_PRINT("tag[%d] = %d\n", i, (int)tag[i]);
    }
    std::cout<<"\n";
}

int main(int argc, char *argv[]) {
    // ===================== 1. 初始化 =====================

    int msg_len = std::stoi(argv[1]);
    int mode = std::stoi(argv[2]);
    int alg_type = std::stoi(argv[3]);
    printf("----------------msg_len = %d, mode = %d, alg = %d ----------------\n", msg_len, mode, alg_type);
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    printf("start\n");
    // ===================== 2. 构造所有输入输出 Tensor =====================
    // 你可以根据算子实际需求修改 shape 和数据
    std::vector<int64_t> shape_scalar = {1};       // 标量用 shape [1]
    std::vector<int64_t> key_shape_data = {16};        // key shape
    std::vector<int64_t> input_shape_data = {msg_len};
    std::vector<int64_t> iv_shape_data = {32};
    std::vector<int64_t> shape_op_cfg = {6};
    std::vector<int64_t> shape_output = {msg_len};      // 输出 shape
    std::vector<int64_t> shape_y = {1};
    std::vector<int64_t> shape_tag = {GCM_TAG_SIZE};
    // 设备地址
    void* keyDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* ivDeviceAddr = nullptr;
    void* opConfigDeviceAddr = nullptr;
    void* tagRefOptionalDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;


    // Tensor 指针
    aclTensor* key = nullptr;
    aclTensor* input = nullptr;
    aclTensor* iv = nullptr;
    aclTensor* opConfig = nullptr;
    aclTensor* tagRefOptional = nullptr;
    aclTensor* output = nullptr;
    aclTensor* y = nullptr;

    // 构造测试数据（全部使用 Tensor）
    std::vector<uint8_t> keyHostData(16, 0x11);            // key 数据
    std::vector<uint8_t> inputHostData(msg_len, 0x00);         // 输入数据

    std::vector<uint8_t> ivHostData(16, 0x1f);            // iv
    std::vector<uint32_t> opConfigHostData = {1, static_cast<uint32_t>(mode), static_cast<uint32_t>(alg_type), 1, 34, 0};          // 配置 version, mode, alg_type, key_type, keyId, device_id
    std::vector<uint8_t> outputHostData(msg_len, 0);           // 输出
    std::vector<uint32_t> yHostData = {0};           // 输出

    std::vector<uint8_t> outExpectedData(msg_len, 0);
    std::vector<uint8_t> tagHostData(16, 0);
    if (alg_type == 1 && mode != 0) {
        HostCrypto(reinterpret_cast<unsigned char*>(keyHostData.data()), reinterpret_cast<unsigned char*>(ivHostData.data()), msg_len, reinterpret_cast<unsigned char*>(inputHostData.data()), reinterpret_cast<unsigned char*>(outExpectedData.data()), mode);
    } else if (alg_type == 2 && mode != 0) {
        HostCryptoGcm(reinterpret_cast<unsigned char*>(keyHostData.data()), reinterpret_cast<unsigned char*>(ivHostData.data()), msg_len, reinterpret_cast<unsigned char*>(inputHostData.data()), reinterpret_cast<unsigned char*>(outExpectedData.data()), reinterpret_cast<unsigned char*>(tagHostData.data()), mode);
    }

    // ===================== 创建所有 aclTensor =====================
    CreateAclTensor(keyHostData, key_shape_data, &keyDeviceAddr, ACL_UINT8, &key);

    CreateAclTensor(inputHostData, input_shape_data, &inputDeviceAddr, ACL_UINT8, &input);

    CreateAclTensor(ivHostData, iv_shape_data, &ivDeviceAddr, ACL_UINT8, &iv);
    CreateAclTensor(opConfigHostData, shape_op_cfg, &opConfigDeviceAddr, ACL_UINT32, &opConfig);
    CreateAclTensor(outputHostData, shape_output, &outputDeviceAddr, ACL_UINT8, &output);
    CreateAclTensor(yHostData, shape_y, &yDeviceAddr, ACL_UINT32, &y);

    CreateAclTensor(tagHostData, shape_tag, &tagRefOptionalDeviceAddr, ACL_UINT8, &tagRefOptional);
    printf("tensor created\n");
    // ===================== 3. 调用自定义算子 aclnnCrypto =====================
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnCryptoGetWorkspaceSize(
        key,
        input,
        output,
        iv,
        opConfig,
        tagRefOptional,
        nullptr,
        y,
        &workspaceSize,
        &executor
    );
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCryptoGetWorkspaceSize failed retCode: %d\n", ret); return ret);
    printf("getworkspaceSIze %d\n", workspaceSize);
    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc workspace failed\n"); return ret);
    }
    printf("malloc workspaceSize\n");
    // 【第二段】执行算子
    ret = aclnnCrypto(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrypto execute failed retCode: %d\n",ret); return ret);

    // ===================== 4. 同步流 =====================
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("stream sync failed %d \n", ret); return ret);

    // ===================== 5. 拷贝输出到主机并打印 =====================
    auto outSize = GetShapeSize(shape_output);
    std::vector<uint8_t> result(outSize);
    ret = aclrtMemcpy(
        result.data(), outSize * sizeof(uint8_t),
        outputDeviceAddr, outSize * sizeof(uint8_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );

    LOG_PRINT("=== aclnnCrypto output result ===\n");
    for (int i = 0; i < 10; i++) {
        LOG_PRINT("output[%d] = %d\n", i, result[i]);
    }
    printf("is device crypto match host crypto: %d, 0 for yes\n", ExpectEQ(reinterpret_cast<unsigned char*>(outExpectedData.data()), reinterpret_cast<unsigned char*>(result.data()), msg_len));

    std::vector<uint32_t> resultY(1);
    ret = aclrtMemcpy(
        resultY.data(), 1 * sizeof(uint32_t),
        yDeviceAddr, 1 * sizeof(uint32_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );

    LOG_PRINT("=== aclnnCrypto y result ===\n");
    for (int i = 0; i < 1; i++) {
        LOG_PRINT("y[%d] = %d\n", i, resultY[i]);
    }
    if (mode == 0) {
        std::vector<uint8_t> resKey(16, 0);
        ret = aclrtMemcpy(
            resKey.data(), resKey.size(),
            keyDeviceAddr, resKey.size(),
            ACL_MEMCPY_DEVICE_TO_HOST
        );

        LOG_PRINT("=== aclnnCrypto Read Key ===\n");
        for (int i = 0; i < resKey.size(); i++) {
            LOG_PRINT("key[%d] = %d\n", i, (int)(resKey[i]));
        }
    }
    if (alg_type == 2) {
        std::vector<uint8_t> resTag(16, 0);
        ret = aclrtMemcpy(
            resTag.data(), resTag.size(),
            tagRefOptionalDeviceAddr, resTag.size(),
            ACL_MEMCPY_DEVICE_TO_HOST
        );

        LOG_PRINT("=== aclnnCrypto tag ===\n");
        for (int i = 0; i < resTag.size(); i++) {
            LOG_PRINT("tag[%d] = %d\n", i, (int)(resTag[i]));
        }
    }

    // ===================== 6. 销毁资源 =====================
    aclDestroyTensor(key);
    aclDestroyTensor(input);
    aclDestroyTensor(iv);
    aclDestroyTensor(opConfig);
    aclDestroyTensor(output);
    aclDestroyTensor(tagRefOptional);

    aclrtFree(keyDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(ivDeviceAddr);
    aclrtFree(opConfigDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(tagRefOptionalDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("aclnnCrypto test success!\n");
    printf("======================================\n");
    return 0;
}
```