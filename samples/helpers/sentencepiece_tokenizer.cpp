//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// SentencePiece Tokenizer                                                     (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <list>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <limits>

//-------------------------------------- Project  Headers ------------------------------------------

#include "sentencepiece_tokenizer.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

using Range = SentencePieceBPETokenizer::Range;
using score_t = SentencePieceBPETokenizer::score_t;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct a new SentencePieceBPETokenizer object
 *
 * @param enc String encoding to use
 */
SentencePieceBPETokenizer::SentencePieceBPETokenizer(encoding enc) : encoding_(enc) {
}


/**
 * @brief Check if a token is a "special" token, like BOS, EOS etc.
 *
 * @param token Index of the token to check
 *
 * @retval true Token is a special token
 * @retval false Token is a regular token
 */
bool SentencePieceBPETokenizer::isSpecialToken(uint32_t token) const {
    if ((token == unknownToken_) || (token == bosToken_) || (token == eosToken_) || (token == padToken_)) {
        return true;
    }
    return false;
}


/**
 * @brief Convert token to string
 *
 * @param token Token index to convert
 * @param pretty If \c true, will remove control codes from the output to have a nicer look
 * @param lineStart If \c false, will prepend a space to the output when a "new word" prefix is
 *                  encountered
 * @return String that is represented by the token, varies in length
 */
std::string SentencePieceBPETokenizer::tokenToString(uint32_t token, bool pretty, bool lineStart) const {
    if (auto it = dictionary_.find(token) ; it != dictionary_.end()) {
        if ((token == bosToken_) || (token == eosToken_) || (token == padToken_) ||
            (token == unknownToken_)) return {};
        if (!pretty) return it->second.data;
        else {
            auto * utf8 = (const uint8_t *)it->second.data.c_str();
            if (utf8[0] == 0xE2 && utf8[1] == 0x96 && utf8[2] == 0x81) {
                if (it->second.data.size() == 3) {
                    return (lineStart) ? std::string() : std::string(" ");
                }
                else {
                    std::string test = it->second.data.substr(3);
                    return (lineStart) ? it->second.data.substr(3) : std::string(" ") + it->second.data.substr(3);
                }
            } else return it->second.data;
        }
    } else return {};
}


/**
 * @brief Tokenize an input string
 *
 * @param text Input string that is subject to tokenization
 * @param start When \c true, will prepend a start-of-stream token to the output
 *
 * @return Vector of tokens representing the input string
 *
 * This function splits the supplied input string into a vector of tokens which are suitable as
 * input for a transformer network.
 */
std::vector<uint32_t> SentencePieceBPETokenizer::tokenize(const std::string &text, bool start) const {
    std::string normalized = normalize(text, true);
    auto symbols = (encoding_ == UTF8) ? splitUTF8(normalized) : splitLatin1(normalized);
    //----------------------------------------------
    // Greedy merging of symbols to fit into tokens
    //----------------------------------------------
    while (true) {
        score_t best = -std::numeric_limits<score_t>::max();
        score_t secondbest = -std::numeric_limits<score_t>::max();
        auto bestit = symbols.begin();
        auto lastit = bestit;
        int besttoken = -1;
        for (auto it = ++(symbols.begin()); it != symbols.end(); ++it) {
            auto sc = getScore((*lastit) + (*it), normalized);
            score_t current = sc.first;
            if (current > best) {
                secondbest = best;
                best = current;
                bestit = lastit;
                besttoken = sc.second;
            }
            lastit = it;
        }
        if (best != -std::numeric_limits<score_t>::max()) {
            auto target = bestit;
            *target = (*target).merge(*(++bestit), best, besttoken);
            symbols.erase(bestit);
            score_t bwdmatch, fwdmatch;
            do {
                bwdmatch = -std::numeric_limits<score_t>::max();
                fwdmatch = -std::numeric_limits<score_t>::max();
                int fwdtoken = -1;
                int bwdtoken = -1;
                auto prev = target;
                if (target != symbols.begin()) {
                    auto sc = getScore(*(--prev) + (*target), normalized);
                    bwdmatch = sc.first;
                    bwdtoken = sc.second;
                }
                auto next = target;
                if (++next != symbols.end()) {
                    auto sc = getScore((*target) + (*next), normalized);
                    fwdmatch = sc.first;
                    fwdtoken = sc.second;
                }
                if (std::max(fwdmatch, bwdmatch) > secondbest && std::max(fwdmatch, bwdmatch) > -std::numeric_limits<score_t>::max()) {
                    if (fwdmatch > bwdmatch) {
                        *target = (*target).merge(*next, fwdmatch, fwdtoken);
                        symbols.erase(next);
                    } else {
                        *prev = (*prev).merge(*target, bwdmatch, bwdtoken);
                        symbols.erase(target);
                        target = prev;
                    }
                }
            } while (std::max(bwdmatch, fwdmatch) > secondbest);
        } else {
            break;
        }
    }
    //----------------------------------------------
    // Now check for leftovers and tokenize those
    //----------------------------------------------
    for (auto it = symbols.begin(); it != symbols.end(); ++it) {
        if (it->score == -std::numeric_limits<score_t>::max()) {
            auto sc = getScore(*it, normalized);
            it->score = sc.first;
            it->token = sc.second;
        }
    }
    std::vector<uint32_t> tokens;
    tokens.reserve(symbols.size() + ((start) ? 1 : 0));
    if (start) tokens.emplace_back(bosToken_);
    for (auto sym: symbols) tokens.emplace_back((uint32_t)sym.token);
    return tokens;
}


/**
 * @brief Load a tokenizer model from a .model file
 *
 * @param filename Filename of the tokenizer model file (stored by the SentencePiece tokenizer)
 *
 * @return Number of tokens in the vocabulary
 *
 * This function is a hack that parses the vocabulary file written by SentencePiece (using protobuf)
 * and extracts the vocabulary. As I did not want to add a dependency on protobuf and/or distribute
 * the protobuf-generated code, I decided to parse the file manually. Caveat emptor.
 */
int SentencePieceBPETokenizer::loadVocabulary(const std::string& filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot load tokenizer model file " + filename);
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
    if (fread(buffer.get(), 1, size, f) != size) throw std::runtime_error("Cannot load tokenizer model file " + filename);
    fclose(f);
    size_t offset = 0;
    int tokenidx = 0;
    while (offset < size) {
        uint8_t tag = buffer[offset++];
        if ((tag != 0xA) && (tag != 0x12)) throw std::runtime_error("Unknown tokenizer model file format in " + filename);
        // read field length
        uint32_t fieldlen = buffer[offset] & 0x7F;
        int shift = 7;
        while (buffer[offset] & 0x80) {fieldlen |= (buffer[++offset] & 0x7F) << shift; shift += 7;}
        offset++;
        if (tag == 0x12) {
            bool ok = parsePostamble(buffer.get(), (int)offset, (int)size);
            return (ok) ? tokenidx : 0;
        }
        if (tag != 0xA) {
            offset += fieldlen;
            continue;
        }
        // read token-string length
        if (buffer[offset++] != 0xA) {
            throw std::runtime_error("Unknown tokenizer model file format in " + filename);
        }
        uint32_t tokenlen = buffer[offset] & 0x7F;
        shift = 7;
        while (buffer[offset] & 0x80) { tokenlen |= (buffer[++offset] & 0x7F) << shift ; shift += 7;}
        offset++;
        uint8_t *tokenptr = buffer.get() + offset;
        offset += tokenlen;
        if (buffer[offset++] != 0x15) throw std::runtime_error("Unknown tokenizer model file format in " + filename);
        score_t score;
        // NOTE (mw) this presumes that we are running on a little-endian architecture
        memcpy(&score, buffer.get() + offset, sizeof(score));
        offset += sizeof(score);
        uint8_t type = 0x1;
        if (buffer[offset++] != 0x18) {
            if ((buffer[offset-1] != 0xA) && (buffer[offset-1] != 0x12)) throw std::runtime_error("Unknown tokenizer model file format in " + filename);
            offset--;
        } else {
            type = buffer[offset++];
        }
        addToken(tokenptr, tokenlen, score, type, tokenidx++);
    }
    return tokenidx;
}


/**
 * @brief Parse the "postamble" of the tokenizer model file
 *
 * @param buffer Pointer to token model buffer
 * @param offset Offset within \p buffer
 * @param size Total size of \p buffer
 *
 * @retval true parsing required information from the postamble was successful
 * @retval false otherwise
 *
 * This sifts through the trailing part of the model file and extracts information about the
 * "special" tokens.
 */
bool SentencePieceBPETokenizer::parsePostamble(const uint8_t *buffer, int offset, int size) {
    uint8_t done = 0;
    while ((offset < size) && (done != 0x7)) {
        uint16_t tag = (uint16_t)buffer[offset++];
        if (buffer[offset-1] >= 128) tag += (buffer[offset++]-1) << 7;
        assert(buffer[offset-1] < 128);  // we do not support longer tags
        uint64_t fieldlen = buffer[offset] & 0x7F;
        if ((tag != 0x55) && (tag != 0x7D)) {
            int shift = 7;
            while (buffer[offset] & 0x80) {
                fieldlen |= (buffer[++offset] & 0x7F) << shift;
                shift += 7;
            }
            offset++;
        }
        switch (tag) {
            case 0x0A:
            case 0x12:
            case 0x3A:
                offset += (int)fieldlen;
                break;
            case 0x55:
            case 0x7D:
                offset += 4;
                break;
            case 0x140:  // unknown token index
                unknownToken_ = (int)fieldlen;
                done |= 1;
                break;
            case 0x148:  // begin-of-stream token index
                bosToken_ = (int)fieldlen;
                done |= 2;
                break;
            case 0x150:  // end-of-stream token index
                eosToken_ = (int)fieldlen;
                done |= 4;
                break;
            case 0x158:  // pad token index
                // ignore for now as we do not support batch-processing anyway
                break;
            default:
                break;
        }
    }
    return (done == 0x7);
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Normalize an input string
 *
 * @param text Input string
 * @param escapeWhitespace If true, white-space characters are escaped
 * @param preserveNewline
 *
 * @return Normalized string
 *
 * This function normalizes an input string by converting it into symbols first, escaping white-space
 * characters (if enabled) and trim the result.
 *
 * @todo This needs to be redone, lots of cases not covered
 */
std::string SentencePieceBPETokenizer::normalize(const std::string& text, bool escapeWhitespace) const {
    auto symbols = (encoding_ == UTF8) ? splitUTF8(text) : splitLatin1(text);
    const std::string whitespace = "\xe2\x96\x81";
    std::string trimmed = trim(symbols, text);
    std::string output;
    output.reserve(3*trimmed.size());
    bool wslead = false;
    if (escapeWhitespace && (!isWhitespace(symbols.front(), trimmed))) {
        wslead = true;
    }
    symbols = (encoding_ == UTF8) ? splitUTF8(trimmed) : splitLatin1(trimmed);
    for (auto sym : symbols) {
        if (isWhitespace(sym, trimmed) && escapeWhitespace) {
            if (isNewline(sym, trimmed)) output += trimmed.substr(sym.start, sym.len());
            wslead = true;
        } else {
            if (wslead) output += whitespace;
            output += trimmed.substr(sym.start, sym.len());
            wslead = false;
        }
    }
    return output;
}


/**
 * @brief Trim a symbol set at both ends by removing leading/trailing white-spaces
 *
 * @param symbols List of symbols to trim
 * @param data Underlying string data
 *
 * @return Trimmed string data
 */
std::string SentencePieceBPETokenizer::trim(std::list<Range> symbols, const std::string& data) {
    auto it = symbols.begin();
    while (it != symbols.end()) {
        if (isWhitespace(*it, data) && !isNewline(*it, data)) it = symbols.erase(it);
        else break;
    }
    while (!symbols.empty() && isWhitespace(symbols.back(), data) && !isNewline(symbols.back(), data)) symbols.pop_back();
    std::string output;
    for (auto sym : symbols) {
        output += data.substr(sym.start, sym.len());
    }
    return output;
}


/**
 * @brief Check if a symbol is a white-space characters
 *
 * @param range Symbol to check for being a white-space character
 * @param data Underlying string data
 *
 * @retval true Symbol is a white-space character
 * @retval false Otherwise
 */
bool SentencePieceBPETokenizer::isWhitespace(const Range& range, const std::string &data) {
    if (range.len() == 1) {
        size_t offset = range.start;
        return (data[offset] == ' ' || data[offset] == '\t' || data[offset] == '\n' || data[offset] == '\r' || data[offset] == '\x85' || data[offset] == '\xa0');
    }
    if (range.len() == 2) {
        size_t offset = range.start;
        uint16_t code = ((data[offset] & 0x7F) << 7) | data[offset+1];
        return (code == 0x1680) || ((code >= 0x2000) && (code <= 0x200A)) || (code == 0x2028) || (code == 0x2029) || (code == 0x202F) || (code == 0x205F) || (code == 0x3000);
    }
    return false;
}


bool SentencePieceBPETokenizer::isNewline(const Range& range, const std::string& data) {
    if (range.len() == 1) {
        size_t offset = range.start;
        return (data[offset] == '\n');
    }
    return false;
}

/**
 * @brief Check if a range of bytes matches a known token in the vocabulary
 *
 * @param range Range to check
 * @param data Underlying string data to the range
 *
 * @retval true if the range matches a known token
 * @retval false otherwise
 */
bool SentencePieceBPETokenizer::isKnown(const Range& range, const std::string& data) const {
    auto sctpair = getScore(range, data);
    return (sctpair.second != -1);
}


/**
 * @brief Obtain score for provided range
 *
 * @param range Range that represents one or more symbols in the input string
 * @param data Underlying string data
 *
 * @return Pair of token score and token index
 *
 * This function checks of the provided data range maps to a known token and returns the score of
 * the token as well as the index of the token in the dictionary. In case the token could not be
 * matched, this function returns a pair of \c -FLT_MAX and \c -1, indicating that no known token
 * exists for that symbol combination.
 */
std::pair<score_t, int> SentencePieceBPETokenizer::getScore(const Range& range, const std::string& data) const {
    uint32_t hashvalue = hash((const uint8_t *)data.c_str() + range.start, range.end - range.start + 1);
    auto it = vocabulary_.find(hashvalue);
    int index = -1;
    score_t score = -std::numeric_limits<float>::max();
    while (it != vocabulary_.end() && it->first == hashvalue) {
        if (it->second.data.size() == range.len() && memcmp(data.c_str() + range.start, it->second.data.c_str(), range.len()) == 0) {
            if (it->second.score == 0) {
                if (score == -std::numeric_limits<float>::max()) {
                    score = it->second.score;
                    index = it->second.index;
                }
            } else {
                if ((it->second.score > score) || (score == 0)) {
                    score = it->second.score;
                    index = it->second.index;
                }
            }
        }
        ++it;
    }
    return {score, index};
}


/**
 * @brief Compute hash over data range
 *
 * @param data Pointer to data to compute hash for
 * @param len Length of data range to compute hash for
 *
 * @return Hash value
 */
uint32_t SentencePieceBPETokenizer::hash(const uint8_t *data, size_t len) {
    uint32_t hash = 5381;
    for (size_t i = 0; i < len; i++) hash = (hash * 351727) ^ (data[i] * 134999);
    return hash;
}


/**
 * @brief Split a UTF-8 string into a list of symbols
 *
 * @param text Input text to split
 *
 * @return List of range, where each range corresponds to one symbol
 */
std::list<Range> SentencePieceBPETokenizer::splitUTF8(const std::string& text) {
    std::list<Range> symbols;
    size_t srcidx = 0;
    size_t rem = text.size();
    auto * ptr = (const uint8_t *)text.c_str();
    while (srcidx < text.size()) {
        int l = std::min(utf8Len(ptr+srcidx), (int)rem);
        symbols.emplace_back(srcidx, srcidx + l-1);
        srcidx += l;
    }
    return symbols;
}


/**
 * @brief Get number of bytes for supplied utf-8 token
 *
 * @param ptr Pointer to utf-8 token
 *
 * @return Number of bytes used to represent that token/character
 */
inline int SentencePieceBPETokenizer::utf8Len(const uint8_t *ptr) {
    static const uint8_t clen[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return clen[ptr[0] >> 4];
}


/**
 * @brief Split a latin-1 string into a list of symbols
 *
 * @param text Input text to split
 *
 * @return List of range, where each range corresponds to one symbol
 */
std::list<Range> SentencePieceBPETokenizer::splitLatin1(const std::string& text) {
    std::list<Range> symbols;
    for (size_t idx = 1; idx < text.size(); idx++) {
        symbols.emplace_back(idx - 1, idx - 1);
    }
    return symbols;
}


/**
 * @brief Add token to vocabulary and dictionary
 *
 * @param token Pointer to string data underlying the token
 * @param tokenlen Size of the token (in bytes)
 * @param score Token score
 * @param type Token type, see long description, 1 indicates a ?? token, 2 indicates the unknown token, 3 indicates
 *             the begin-of-stream/end-of-stream, token, 6 indicates a regular token
 * @param tokenIdx Token index in the dictionary
 *
 * The following types are used in tokens:
 *   1 - multi-byte token
 *   2 - unknown token
 *   3 - start/stop token
 *   6 - single-byte token
 */
void SentencePieceBPETokenizer::addToken(const uint8_t *token, size_t tokenlen, score_t score, uint8_t type, int tokenIdx) {
    if (tokenlen == 0) return;
    if (type == 6) {
        int value=0;
        uint8_t sbtoken[2]={0};
        sscanf((const char *)token,"<%x>", &value);     // FIXME (mw) do something better here
        assert(value <= 255);
        sbtoken[0] = (char)value;
        vocabulary_.insert({hash(sbtoken, 1), Token(std::string(sbtoken,sbtoken+1), score, tokenIdx, type)});
        dictionary_.insert({tokenIdx, Token(std::string(sbtoken, sbtoken+1), score, tokenIdx, type)});
    } else {
        vocabulary_.insert({hash(token, tokenlen), Token(std::string(token, token + tokenlen), score, tokenIdx, type)});
        dictionary_.insert({tokenIdx, Token(std::string(token, token + tokenlen), score, tokenIdx, type)});
    }
}

// vim: set expandtab ts=4 sw=4:

