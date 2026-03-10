#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "proto/inferential.pb.h"

namespace inferential {

/// Holds tensor data received from the server.
struct TensorData {
    std::string key;
    std::vector<uint8_t> data;
    std::vector<int64_t> shape;
    ::inferential::DType dtype;

    /// Returns bytes per element for the tensor's dtype.
    [[nodiscard]] size_t element_size() const {
        switch (dtype) {
            case FLOAT16:
            case BFLOAT16:
                return 2;
            case FLOAT32:
            case INT32:
                return 4;
            case FLOAT64:
            case INT64:
                return 8;
            case UINT8:
            case BOOL:
                return 1;
            default:
                return 0;
        }
    }

    /// Returns the total number of elements.
    [[nodiscard]] size_t numel() const {
        if (element_size() == 0) return 0;
        return data.size() / element_size();
    }

    /// Reinterpret the raw bytes as a typed pointer (zero-copy view).
    /// Returns {pointer, count}.
    template <typename T>
    [[nodiscard]] std::pair<const T*, size_t> as() const {
        if (sizeof(T) != element_size()) {
            throw std::invalid_argument(
                "Type size mismatch: sizeof(T)=" + std::to_string(sizeof(T)) +
                " but element_size()=" + std::to_string(element_size()));
        }
        return {reinterpret_cast<const T*>(data.data()), data.size() / sizeof(T)};
    }
};

}  // namespace inferential
