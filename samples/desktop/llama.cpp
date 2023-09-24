//--------------------------------------------------------------------------------------------------
// FyuseNet Samples
//--------------------------------------------------------------------------------------------------
// LLaMa Sample Main Program                                                   (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/llama_4bit.h"
#include "../helpers/sentencepiece_tokenizer.h"
#include "cxxopts.hpp"

#ifdef FYUSENET_USE_GLFW
#include <fyusenet/gl/glcontext.h>
#endif

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------

/**
 * @brief Check generated token sequence for stop tokens and trim answer appropriately
 *
 * @param[inout] tokens Vector of generated token IDs
 * @param stopTokens 2D vector of token or token-sequences that indicate that the ML should terminate
 *                   the answer
 *
 * @retval true if the token sequence suggests that the answer is one
 * @retval false otherwise
 *
 * This checks if the generated token sequence ends with one of the supplied stop tokens or stop-
 * sequences. Usually the <eos> token indicates a complete answer, however since LLMs are basically
 * token predictors and therefore text completion engines, sometimes the network completes the
 * dialog by impersonating the other party, which is undesired behaviour here. In order to avoid
 * that, token sequences that usually indicate that it is the asking party's turn (such as
 * "You: " or "User: ") should be added to the \p stopTokens list. Make sure that in case you have
 * nested sequences for stopping, to put the longest sequence first.
 */
static bool checkForStopTokens(std::vector<uint32_t>& tokens, const std::vector<std::vector<uint32_t>>& stopTokens) {
    for (auto & stop : stopTokens) {
        if (stop.size() >= tokens.size()) continue;
        bool stopme = true;
        int idx = 0;
        for (auto it = tokens.end() - stop.size(); it != tokens.end(); ++it, idx++) {
            if (*it != stop.at(idx)) {
                stopme = false;
                break;
            }
        }
        if (stopme) {
            tokens.erase(tokens.end()-stop.size(), tokens.end());
            return true;
        }
    }
    return false;
}



int main(int argc, char **argv) {
    cxxopts::Options options(argv[0],"Sample LlaMa LLM Chat");
    options.add_options()("h,help","Get program help")
                         ("w,weights", "Use supplied filename as weight file (mandatory)", cxxopts::value<std::string>())
                         ("t,tokenmodel","Use supplied filename as vocabulary for tokenizer (mandatory)", cxxopts::value<std::string>());
    auto opts = options.parse(argc, argv);
    if ((opts.count("help") > 0) || (opts.count("weights") == 0) || (opts.count("tokenmodel") == 0)) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    // -------------------------------------------------------
    // Setup GL context and thread/PBO pool.
    // -------------------------------------------------------
    auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
    if (!glmgr) {
        std::cerr<<"Cannot setup GL context\n";
        return 1;
    }
    fyusion::fyusenet::GfxContextLink ctx = glmgr->createMainContext();
#ifdef FYUSENET_MULTITHREADING
    fyusion::opengl::AsyncPool::setMaxGLThreads(4);
#endif
    glmgr->setupPBOPools(2, 2);
    glmgr->setupTexturePool();
    // -------------------------------------------------------
    // Instantiate network
    // -------------------------------------------------------
    auto * net = new LlaMa4Bit(ctx);
    printf("Loading model....(may take a bit)\n");fflush(stdout);
    net->useParameterFile(opts["weights"].as<std::string>());
    net->setup();
#ifdef FYUSENET_USE_GLFW
    static bool buttonup = false;
    auto * glctx = dynamic_cast<const fyusion::opengl::GLContext *>(ctx.interface());
    auto mousecb = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) buttonup = true;
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecb);
    while (!buttonup) {
        glfwWaitEventsTimeout(0.1);
    }
    for (int i=0; i<4; i++) {
        glctx->sync();
    }
#endif
    // -------------------------------------------------------
    // Setup tokenizer
    // -------------------------------------------------------
    SentencePieceBPETokenizer tokenizer(SentencePieceBPETokenizer::UTF8);
    int vocabsize = tokenizer.loadVocabulary(opts["tokenmodel"].as<std::string>());
    if (vocabsize <= 0) {
        std::cerr<<"Cannot load vocabulary\n";
        return 1;
    }
    // -------------------------------------------------------
    // Set stop token combinations...
    // -------------------------------------------------------
    std::vector<std::vector<uint32_t>> stoptokens;
    stoptokens.emplace_back(std::vector<uint32_t>{887, 29901});
    stoptokens.emplace_back(std::vector<uint32_t>{tokenizer.stopToken()});
    // -------------------------------------------------------
    // Run a small example chat...
    // -------------------------------------------------------
    std::unique_ptr<uint32_t[]> input(new uint32_t[net->maxSequenceLen()]);
    memset(input.get(), 0, net->maxSequenceLen()*sizeof(uint32_t));
    uint32_t * tokenptr = input.get();
    std::string context("This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.");
    std::cout<<context<<"\n";
    std::cout<<"Assistant: Hello, how may I help you ?\n"<<std::flush;
    bool initial = true, chatty=true;
    std::vector<uint32_t> alltokens;
    // -------------------------------------------------------
    // Main chat-loop
    // -------------------------------------------------------
    while (chatty) {
        [[maybe_unused]] char * discard;
        char querybuffer[2048] = {0};
        discard = fgets(querybuffer, sizeof(querybuffer) - 1, stdin);
        std::string query(querybuffer);
        std::string prefixedquery = std::string("\nYou: ") + query;
        if (initial) {
            prefixedquery = context + prefixedquery;
            initial = false;
        }
        prefixedquery += "Assistant: ";
        // -------------------------------------------------------
        // Tokenize the query (user text) and feed it into the
        // network...
        // -------------------------------------------------------
        auto querytokens = tokenizer.tokenize(prefixedquery, initial);
        std::copy(querytokens.begin(), querytokens.end(), tokenptr);
        net->setInputTokens(input.get(), (int)querytokens.size());
        auto * state = new fyusion::fyusenet::StateToken();
        state->seqLength = (int)querytokens.size();
        state->seqIndex = (int)alltokens.size();
        alltokens.insert(alltokens.end(), querytokens.begin(), querytokens.end());
        net->forward(state);
        // -------------------------------------------------------
        // Get the predicted token and feed it back into the
        // network until we get a stop token (sequence) or are
        // out of token space...
        // -------------------------------------------------------
        uint32_t token = net->getPredictedToken();
        std::vector<uint32_t> answer{token};
        std::string lastout;
        std::vector<std::string> response{tokenizer.tokenToString(token, true, true)};
        int respidx = 0;
        while (!checkForStopTokens(answer, stoptokens) && (int)alltokens.size() < net->maxSequenceLen()) {
            if (response.size() > 2) std::cout<<response[respidx++]<<std::flush;  // give the token prediction a bit of a headstart to cut impostor tokens
            net->rotateInputToken();
            state->seqIndex += state->seqLength;
            state->seqLength = 1;
            net->forward(state);
            token = net->getPredictedToken();
            answer.emplace_back(token);
            response.emplace_back(tokenizer.tokenToString(token, true, false));
        }
        response.resize(answer.size());
        while (respidx < (int)response.size()) std::cout<<response[respidx++]<<std::flush;
        alltokens.insert(alltokens.end(), answer.begin(), answer.end());
        if (response.back().back() != '\n') std::cout<<"\n"<<std::flush;
        if ((int)alltokens.size() >= net->maxSequenceLen()-1) chatty = false;
        delete state;
    }
    // -------------------------------------------------------
    // If we use GLFW, wait for another MB click before
    // terminating
    // -------------------------------------------------------
#ifdef FYUSENET_USE_GLFW
    static bool buttondown = false;
    glctx->sync();
    auto mousecbout = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) {
            buttondown = true;
        }
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecbout);
    while (!buttondown) {
        glfwWaitEventsTimeout(0.1);
    }
#endif
    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    net->cleanup();
    delete net;
    ctx.reset();
    glmgr->tearDown();
    return 0;
}
