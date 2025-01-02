// base/io-funcs.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/io-funcs.h"
#include "base/kaldi-math.h"

namespace kaldi {

// 将布尔值b写入到输出流os中
template <>
void WriteBasicType<bool>(std::ostream& os, bool binary, bool b) {
  os << (b ? "T" : "F");
  if (!binary) os << " ";
  if (os.fail()) KALDI_ERR << "Write failure in WriteBasicType<bool>";
}

// 从输入流 is 中读取一个布尔值
template <>
void ReadBasicType<bool>(std::istream& is, bool binary, bool* b) {
  KALDI_PARANOID_ASSERT(b != NULL);
  if (!binary) is >> std::ws;  // eat up whitespace.
  char c = is.peek();
  if (c == 'T') {
    *b = true;
    is.get();
  } else if (c == 'F') {
    *b = false;
    is.get();
  } else {
    KALDI_ERR << "Read failure in ReadBasicType<bool>, file position is "
              << is.tellg() << ", next char is " << CharToString(c);
  }
}

// 将浮点数float写入到输出流std::ostream中。
template <>
void WriteBasicType<float>(std::ostream& os, bool binary, float f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char*>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

// 将double写入输出流td::ostream
template <>
void WriteBasicType<double>(std::ostream& os, bool binary, double f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char*>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

// 将float从istream流中读出来
template <>
void ReadBasicType<float>(std::istream& is, bool binary, float* f) {
  KALDI_PARANOID_ASSERT(f != NULL);
  if (binary) {
    double d;
    int c = is.peek();
    if (c == sizeof(*f)) {
      is.get();
      is.read(reinterpret_cast<char*>(f), sizeof(*f));
    } else if (c == sizeof(d)) {
      ReadBasicType(is, binary, &d);
      *f = d;
    } else {
      KALDI_ERR << "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *f;
  }
  if (is.fail()) {
    KALDI_ERR << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

// 将double从istream流中读出来
template <>
void ReadBasicType<double>(std::istream& is, bool binary, double* d) {
  KALDI_PARANOID_ASSERT(d != NULL);
  if (binary) {
    float f;
    int c = is.peek();
    if (c == sizeof(*d)) {
      is.get();
      is.read(reinterpret_cast<char*>(d), sizeof(*d));
    } else if (c == sizeof(f)) {
      ReadBasicType(is, binary, &f);
      *d = f;
    } else {
      KALDI_ERR << "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *d;
  }
  if (is.fail()) {
    KALDI_ERR << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

// 检查一个字符串是否为空或包含空白字符
void CheckToken(const char* token) {
  if (*token == '\0') KALDI_ERR << "Token is empty (not a valid token)";
  const char* orig_token = token;
  while (*token != '\0') {
    if (::isspace(*token))
      KALDI_ERR << "Token is not a valid token (contains space): '"
                << orig_token << "'";
    token++;
  }
}

// 将一个字符串token写入到输出流os中
void WriteToken(std::ostream& os, bool binary, const char* token) {
  // binary mode is ignored;
  // we use space as termination character in either case.
  KALDI_ASSERT(token != NULL);
  CheckToken(token);  // make sure it's valid (can be read back)
  os << token << " ";
  if (os.fail()) {
    KALDI_ERR << "Write failure in WriteToken.";
  }
}

// 从输入流中读取下一个字符而不提取它
int Peek(std::istream& is, bool binary) {
  if (!binary) is >> std::ws;  // eat up whitespace.
  return is.peek();
}

// 将一个字符串token写入到输出流os中string版
void WriteToken(std::ostream& os, bool binary, const std::string& token) {
  WriteToken(os, binary, token.c_str());
}

// 从输入流 is 中读取一个字符串，并进行一些错误检查。
void ReadToken(std::istream& is, bool binary, std::string* str) {
  KALDI_ASSERT(str != NULL);
  if (!binary) is >> std::ws;  // consume whitespace.
  is >> *str;
  if (is.fail()) {
    KALDI_ERR << "ReadToken, failed to read token at file position "
              << is.tellg();
  }
  if (!isspace(is.peek())) {
    KALDI_ERR << "ReadToken, expected space after token, saw instead "
              << CharToString(static_cast<char>(is.peek()))
              << ", at file position " << is.tellg();
  }
  is.get();  // consume the space.
}

// 从输入流 is 中读取一个字符，并根据是否读取到括号 > 来决定是否回退该字符。
int PeekToken(std::istream& is, bool binary) {
  if (!binary) is >> std::ws;  // consume whitespace.
  bool read_bracket;
  if (static_cast<char>(is.peek()) == '<') {
    read_bracket = true;
    is.get();
  } else {
    read_bracket = false;
  }
  int ans = is.peek();
  if (read_bracket) {
    if (!is.unget()) {
      // Clear the bad bit. This code can be (and is in fact) reached, since the
      // C++ standard does not guarantee that a call to unget() must succeed.
      is.clear();
    }
  }
  return ans;
}

// 从输入流 is 中读取一个预期的标记（token），并检查它是否与给定的 token
// 匹配。如果读取的标记与预期的标记不匹配，则会抛出一个错误。
void ExpectToken(std::istream& is, bool binary, const char* token) {
  int pos_at_start = is.tellg();
  KALDI_ASSERT(token != NULL);
  CheckToken(token);           // make sure it's valid (can be read back)
  if (!binary) is >> std::ws;  // consume whitespace.
  std::string str;
  is >> str;
  is.get();  // consume the space.
  if (is.fail()) {
    KALDI_ERR << "Failed to read token [started at file position "
              << pos_at_start << "], expected " << token;
  }
  // The second half of the '&&' expression below is so that if we're expecting
  // "<Foo>", we will accept "Foo>" instead.  This is so that the model-reading
  // code will tolerate errors in PeekToken where is.unget() failed; search for
  // is.clear() in PeekToken() for an explanation.
  if (strcmp(str.c_str(), token) != 0 &&
      !(token[0] == '<' && strcmp(str.c_str(), token + 1) == 0)) {
    KALDI_ERR << "Expected token \"" << token << "\", got instead \"" << str
              << "\".";
  }
}

// 从输入流 is 中读取一个预期的标记（token），并检查它是否与给定的 token
// 匹配。如果读取的标记与预期的标记不匹配，则会抛出一个错误。string版
void ExpectToken(std::istream& is, bool binary, const std::string& token) {
  ExpectToken(is, binary, token.c_str());
}

}  // end namespace kaldi
