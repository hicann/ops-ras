# aclnnCrypto

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T01:46:18.382Z pushedAt=2026-07-30T03:35:18.412Z -->

## Applicable Products

| Product                                                    | Supported |
| :----------------------------------------------------------- | :-------: |
| Ascend 950PR/Ascend 950DT  |     √     |
| Atlas A3 training products/Atlas A3 inference products       |     √     |
| Atlas A2 training products/Atlas A2 inference products       |     √     |
| Atlas 200I/500 A2 inference products                         |     √     |
| Atlas inference products                                     |     √     |
| Atlas training products                                      |     ×     |

## Function

- Description: Includes a key reading function and an encryption/decryption function. The key reading function obtains the SPDM-derived encryption/decryption key and writes it to a device-side tensor. The encryption/decryption function uses the read key to encrypt or decrypt a device-side tensor.

  Key reading: Before reading the key, the host and device have already derived the encryption/decryption key through SPDM, and the caller has obtained the keyId of the key to be read. When reading the key, the key is obtained from KMS-proxy via UDS by key_id, alg_type, and key_type, and written to the device-side aclTensor key.

  Encryption/decryption: During encryption/decryption, standard AES_CTR_128 and AES_GCM_128 algorithms are used for computation. The key used is the passed device-side aclTensor key (already written by the key reading function). The iv and tag used can be passed from the host to the device side.

## Function Prototype

Each operator uses a two-phase API. You must first call the "aclnnCryptoGetWorkspaceSize" API to obtain the required workspace size and the executor that includes the operator computation flow, and then call the "aclnnCrypto" API to perform the computation.

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

- **Parameters**

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>key (aclTensor*)</td>
        <td>Input</td>
        <td>The key used by the operator.</td>
        <td>Does not support null tensor.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputText (aclTensor*)</td>
        <td>Input</td>
        <td>Specifies the plaintext or ciphertext input for operator encryption/decryption.</td>
        <td>Supports null tensor. Nullable when reading key. The shape and type must be consistent with outputText.</td>
        <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BFLOAT16</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputText(aclTensor*)</td>
        <td>Input/Output</td>
        <td>Specifies the plaintext or ciphertext output by the operator encryption/decryption.</td>
        <td>Supports null tensor. Nullable when reading key. The shape and type must be consistent with inputText.</td>
        <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BFLOAT16</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>iv (aclTensor*)</td>
        <td>Input</td>
        <td>IV used for operator encryption/decryption.</td>
        <td>Supports null tensor. Nullable when reading key.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>opConfig (aclTensor*)</td>
        <td>Input</td>
        <td>Operator parameter configuration, structured as {version, mode, alg_type, key_type, key_id, device_id}.</td>
        <td>Does not support null tensor. version indicates the operator version, with a value range of [1,2], where 1 indicates the version supporting A2/A3 and 2 indicates the version supporting A5. mode indicates the operator working mode, with a value range of [0,2], where 0 indicates key reading, 1 indicates encryption mode, and 2 indicates decryption mode. alg_type indicates the algorithm type, with a value range of [1,2], where 1 indicates the AES_CTR_128 algorithm and 2 indicates the AES_GCM_128 algorithm. key_type indicates the key type, with a value range of [1] and a default value of 1, indicating that the key is stored in the tensor pointed to by key. key_id indicates the key ID, with a value range of [0, UINT32_MAX], starting from 0, where a larger key_id indicates a newer key. device_id indicates the device ID, with a value range of [0,63], starting from 0.</td>
        <td>UINT32</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>tag (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Tag used for encryption/decryption.</td>
        <td>Supports null tensor. Nullable when reading key and during AES-CTR-128 encryption/decryption.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>√</td>
      </tr>
      <tr>
        <td>aad (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Auxiliary information used by the operator.</td>
        <td>Supports only null tensor. Passing a non-null tensor is not supported.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>Arbitrary Dimension</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Status code that indicates whether the operator execution succeeds or fails.</td>
        <td>Does not support null tensor.</td>
        <td>UINT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)*</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to apply on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed pointer is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of the operator input is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnCrypto

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained from the first-phase aclnnCryptoGetWorkspaceSize API.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Returns the operator executor, which includes the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

aclnnCrypto defaults to a deterministic implementation.

Input Constraints:

1. When opConfig.mode = 0 (read key), inputText, outputText, iv, and tag can be null (no error occurs if they are not null but their types and shapes comply with the constraints).

  2. key, opConfig, and out cannot be null.

  3. When inputText and outputText are not null, their shapes and sizes must be identical (same type, same number of dimensions, and same size in each dimension).

  4. Non-null input parameters must conform to the type constraints in the table.

  5. When opConfig.algType = 1 (AES-CTR-128 algorithm), tag can be null (if not null, this parameter is not used).

  6. When opConfig.algType = 2 (AES-GCM-128 algorithm), tag cannot be null.

  7. The parameters of opConfig must comply with the definition: version takes values 1 and 2, mode takes values 0, 1, and 2, algType takes values 1 and 2, keyType takes value 1, keyId takes values in [0, UINT32_MAX], and deviceId takes values in [0, 63].

  8. When reading the key, a correct keyId must be passed in (one keyId corresponds to one SPDM-derived encryption/decryption key). Otherwise, an error is reported.

## Example

The sample code is as follows for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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

    // 1. Create and initialize the encryption context.
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;

    // 2. Initialize the encryption algorithm: AES-128-CTR.
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }

    // 3. Encrypt in blocks (supports data of arbitrary length).
    if (1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;

    // 4. Process the last block (no actual data in CTR mode, but must be called).
    if (1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;

    // 5. Release the context.
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

    // 1. Create and initialize the decryption context.
    if (!(ctx = EVP_CIPHER_CTX_new())) return -1;

    // 2. Initialize the decryption algorithm: AES-128-CTR.
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, key, iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }

    // 3. Decrypt in blocks.
    if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;

    // 4. Process the last block (no actual data in CTR mode, but must be called).
    if (1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;
    // 5. Release the context.
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
    // Release.
    EVP_CIPHER_CTX_free(ctx);
    // GCM ciphertext length = plaintext length.
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
    else if (mode == 2) { // Encrypt first, then decrypt.
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
    // ===================== 1. Initialization =====================

    int msg_len = std::stoi(argv[1]);
    int mode = std::stoi(argv[2]);
    int alg_type = std::stoi(argv[3]);
    printf("----------------msg_len = %d, mode = %d, alg = %d ----------------\n", msg_len, mode, alg_type);
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    printf("start\n");
    // ===================== 2. Construct all input and output tensors =====================
    // Modify the shape and data based on actual operator requirements.
    std::vector<int64_t> shape_scalar = {1};       // Use shape [1] for scalars.
    std::vector<int64_t> key_shape_data = {16};        // key shape
    std::vector<int64_t> input_shape_data = {msg_len};
    std::vector<int64_t> iv_shape_data = {32};
    std::vector<int64_t> shape_op_cfg = {6};
    std::vector<int64_t> shape_output = {msg_len};      // Output shape.
    std::vector<int64_t> shape_y = {1};
    std::vector<int64_t> shape_tag = {GCM_TAG_SIZE};
    // Device address.
    void* keyDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* ivDeviceAddr = nullptr;
    void* opConfigDeviceAddr = nullptr;
    void* tagRefOptionalDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;


    // Tensor pointer.
    aclTensor* key = nullptr;
    aclTensor* input = nullptr;
    aclTensor* iv = nullptr;
    aclTensor* opConfig = nullptr;
    aclTensor* tagRefOptional = nullptr;
    aclTensor* output = nullptr;
    aclTensor* y = nullptr;

    // Construct test data (all using Tensor).
    std::vector<uint8_t> keyHostData(16, 0x11);            // Key data.
    std::vector<uint8_t> inputHostData(msg_len, 0x00);         // Input data.

    std::vector<uint8_t> ivHostData(16, 0x1f);            // iv
    std::vector<uint32_t> opConfigHostData = {1, static_cast<uint32_t>(mode), static_cast<uint32_t>(alg_type), 1, 34, 0};          // Configure version, mode, alg_type, key_type, keyId, and device_id.
    std::vector<uint8_t> outputHostData(msg_len, 0);           // Output.
    std::vector<uint32_t> yHostData = {0};           // Output.

    std::vector<uint8_t> outExpectedData(msg_len, 0);
    std::vector<uint8_t> tagHostData(16, 0);
    if (alg_type == 1 && mode != 0) {
        HostCrypto(reinterpret_cast<unsigned char*>(keyHostData.data()), reinterpret_cast<unsigned char*>(ivHostData.data()), msg_len, reinterpret_cast<unsigned char*>(inputHostData.data()), reinterpret_cast<unsigned char*>(outExpectedData.data()), mode);
    } else if (alg_type == 2 && mode != 0) {
        HostCryptoGcm(reinterpret_cast<unsigned char*>(keyHostData.data()), reinterpret_cast<unsigned char*>(ivHostData.data()), msg_len, reinterpret_cast<unsigned char*>(inputHostData.data()), reinterpret_cast<unsigned char*>(outExpectedData.data()), reinterpret_cast<unsigned char*>(tagHostData.data()), mode);
    }

    // ===================== Create all aclTensors =====================
    CreateAclTensor(keyHostData, key_shape_data, &keyDeviceAddr, ACL_UINT8, &key);

    CreateAclTensor(inputHostData, input_shape_data, &inputDeviceAddr, ACL_UINT8, &input);

    CreateAclTensor(ivHostData, iv_shape_data, &ivDeviceAddr, ACL_UINT8, &iv);
    CreateAclTensor(opConfigHostData, shape_op_cfg, &opConfigDeviceAddr, ACL_UINT32, &opConfig);
    CreateAclTensor(outputHostData, shape_output, &outputDeviceAddr, ACL_UINT8, &output);
    CreateAclTensor(yHostData, shape_y, &yDeviceAddr, ACL_UINT32, &y);

    CreateAclTensor(tagHostData, shape_tag, &tagRefOptionalDeviceAddr, ACL_UINT8, &tagRefOptional);
    printf("tensor created\n");
    // ===================== 3. Call the custom operator aclnnCrypto. =====================
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
    // Allocate workspace.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc workspace failed\n"); return ret);
    }
    printf("malloc workspaceSize\n");
    // [Second Phase] Execute the operator.
    ret = aclnnCrypto(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrypto execute failed retCode: %d\n",ret); return ret);

    // ===================== 4. Synchronize the stream. =====================
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("stream sync failed %d \n", ret); return ret);

    // ===================== 5. Copy the output to the host and print. =====================
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

    // ===================== 6. Destroy Resources =====================
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