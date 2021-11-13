/*
 * Copyright 2021 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef wasm_support_entropy_h
#define wasm_support_entropy_h

namespace wasm {

namespace Entropy {

// Estimate how compressible some data is. This is a rough estimate for how
// much smaller the data could get when compressed by something like gzip or
// brotli, but it is a rough estimate that does not necessarily correspond
// perfectly to any existing compression algorithm. The result is a ratio
// compared to the original size, that is, 1.0 indicates we cannot compress and
// we expect to remain at the same size as originally.
double estimateCompressedRatio(const std::vector<uint8_t>& data);

} // namespace Entropy

} // namespace wasm

#endif // wasm_support_entropy_h
