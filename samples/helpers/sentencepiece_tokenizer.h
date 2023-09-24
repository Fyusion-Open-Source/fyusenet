//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// SentencePiece Tokenizer (Header)                                            (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief SentencePiece Byte-Pair-Encoding Tokenizer / Detokenizer
 *
 * This class implements a simple and kind of hacky version of a SentencePiece BPE tokenizer. In
 * the current form it only serves as supplementary material for a sample network in FyuseNet
 * and is not really engineered very well (or at all). It hacks around the requirement of using
 * protobuf to parse the original SentencePiece model file, which makes it very vulnerable to
 * any change in the protobuf part of SentencePiece.
 */
class SentencePieceBPETokenizer {
 public:
    constexpr static uint32_t INVALID_TOKEN = 0xFFFFFFFF;

    /**
     * @brief Encoding type for string data
     */
    enum encoding {
        UTF8,           //!< Use UTF-8 encoding for strings
        LATIN1          //!< Use latin-1 (ISO-8859-1) encoding for strings (not used)
    };

    using score_t = float;

    /**
     * @brief Structure to represent single token and its score for tokenization
     */
    struct Token {
        Token(std::string data, score_t score, int index, uint8_t type) :
            data(std::move(data)), score(score), index(index), type(type) {
        }
        std::string data;   //!< String data that is compounded by the token
        score_t score;      //!< Score of that token, higher scores take precedence in a greedy tokenization scheme
        int index;          //!<
        uint8_t type;
    };

    /**
     * @brief Structure to represent a range of bytes in a string which may be mapped to a token
     */
    struct Range {

        /**
         * @broef Create range with a start / end index in the string
         * @param s Start index
         * @param e End index (inclusive)
         */
        Range(size_t s, size_t e) : start(s), end(e), score(-std::numeric_limits<score_t>::max()), token(-1) {
        }

        /**
         * @brief Concatenate two ranges, assuming that the provided range is \b after the current range
         *
         * @param other Range to concatenate to the current one from the right
         *
         * @return Concatenated range
         */
        Range operator+(const Range& other) const {
            return {start, other.end};
        }

        /**
         * @brief Merge to ranges to a new range, mapping it to a token and equipping it with a score
         *
         * @param other Range to concatenate to the current one from the right
         * @param score Score for the new range
         * @param token Token index for this range
         *
         * @return New (merged) range
         */
        [[nodiscard]] Range merge(const Range& other, score_t score, int token) const {
            Range result(start, other.end);
            result.score = score;
            result.token = token;
            return result;
        }

        /**
         * @brief Retrieve length of range
         *
         * @return Range length (in bytes)
         */
        [[nodiscard]] size_t len() const { return end-start+1;}

        size_t start;       //!< Byte-index into string to be tokenized at start position
        size_t end;         //!< Byte-index into string to be tokenized at end position (inclusive)
        score_t score;      //!< Score for this range in case it constitutes a token
        int token;          //!< Index of the related token in the dictionary
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit SentencePieceBPETokenizer(encoding enc);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<uint32_t> tokenize(const std::string & text, bool start=false) const;
    int loadVocabulary(const std::string& filename);
    [[nodiscard]] std::string tokenToString(uint32_t token, bool pretty=false, bool lineStart=false) const;
    [[nodiscard]] bool isSpecialToken(uint32_t token) const;

    [[nodiscard]] uint32_t stopToken() const {
        return eosToken_;
    }

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] bool isKnown(const Range& range, const std::string& data) const;
    [[nodiscard]] std::pair<score_t,int> getScore(const Range& range, const std::string& data) const;
    [[nodiscard]] static std::list<Range> splitUTF8(const std::string& text);
    [[nodiscard]] static int utf8Len(const uint8_t *ptr);
    [[nodiscard]] static std::list<Range> splitLatin1(const std::string& text);
    [[nodiscard]] std::string normalize(const std::string& text, bool escapeWhitespace) const;
    [[nodiscard]] static std::string trim(std::list<Range> symbols, const std::string& data);
    [[nodiscard]] static bool isWhitespace(const Range& range, const std::string &data);
    [[nodiscard]] static bool isNewline(const Range& range, const std::string& data);
    void addToken(const uint8_t *token, size_t tokenlen, score_t score, uint8_t type, int tokenIdx);
    bool parsePostamble(const uint8_t *buffer, int offset, int size);
    static uint32_t hash(const uint8_t *data, size_t len);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    encoding encoding_ = UTF8;               //!< String encoding that is used on input/output strings (defaults to UTF-8)
    uint32_t unknownToken_ = INVALID_TOKEN;  //!< Index of the special "unknown" token
    uint32_t bosToken_ = INVALID_TOKEN;      //!< Index of the "beginning of stream" token, each sequence starts with it
    uint32_t eosToken_ = INVALID_TOKEN;      //!< Index of the "end of stream" token, each sequence terminates with it
    uint32_t padToken_ = INVALID_TOKEN;      //!< Index of a padding token for batch processing of sequences (which we don't do in FyuseNet)

    /**
     * @brief Vocabulary which maps a token \b hash to an actual token.
     *
     * The hash is computed in the hash() function based on the UTF-8 encoded string data of the
     * respective token.
     */
    std::unordered_multimap<uint32_t, Token> vocabulary_;

    /**
     * Dictionary that maps an actual token index to its token
     */
    std::unordered_multimap<uint32_t, Token> dictionary_;
};


// vim: set expandtab ts=4 sw=4:

