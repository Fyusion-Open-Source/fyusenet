//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Arithmetic Layers Unit Tests
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cmath>
#include <fstream>
#include <atomic>
#include <memory>
#include <thread>

//-------------------------------------- Project  Headers ------------------------------------------

#include <gtest/gtest.h>
#include <fyusenet/fyusenet.h>
#include "gltesthelpers.h"
#include <fyusenet/gpu/singleton_arithlayer.h>
#include <fyusenet/base/layerfactory.h>
#include <fyusenet/gpu/addsublayer.h>
#include "layertestbase.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

using namespace fyusion::fyusenet;
using namespace fyusion::fyusenet::gpu;

int main(int argc,char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GLEnvironment::init();
    return RUN_ALL_TESTS();
}

class ArithLayerTest: public ::testing::Test, public TestContextManager, public LayerTestBase {
 protected:
    ArithLayerTest() : TestContextManager() {
    }

    void SetUp() {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    void TearDown() {
        LayerTestBase::cleanup();
        tearDownGLContext();
    }
};

struct ArithParam {
    ArithParam(float op1, float op2, LayerType op, int w, int h, int chan) :
        operand1(op1), operand2(op2), oper(op), width(w), height(h), channels(chan) {
    }
    float operand1;
    float operand2;
    LayerType oper;
    int width;
    int height;
    int channels;
};

struct SingleArithParam {
    SingleArithParam(float op1, float op2, ArithType op, int w, int h, int chan) :
        oper(op), operand1(op1), operand2(op2), width(w), height(h), channels(chan) {
    }
    ArithType oper;
    float operand1;
    float operand2;
    int width;
    int height;
    int channels;
};

class ParamArithLayerTest : public ArithLayerTest, public ::testing::WithParamInterface<ArithParam> {
};

class ParamSingletonLayerTest: public ArithLayerTest, public ::testing::WithParamInterface<SingleArithParam> {
};

//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_P(ParamSingletonLayerTest, SingletonTestShallow) {
    auto param = GetParam();
    gpu::SingletonArithLayerBuilder bld("single", param.oper);
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels).type(LayerType::SINGLETON_ARITH);
    bld.operand(param.operand2);
    gpu::SingletonArithmeticLayer layer(bld, 1);
    std::unique_ptr<float[]> input(generateConstantData(param.operand1, param.channels, param.width, param.height));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    float expect = 0.f;
    switch (param.oper) {
        case ArithType::ADD:
            expect = param.operand1 + param.operand2;
            break;
        case ArithType::SUB:
            expect = param.operand1 - param.operand2;
            break;
        case ArithType::MUL:
            expect = param.operand1 * param.operand2;
            break;
        case ArithType::DIV:
            expect = param.operand1 / param.operand2;
            break;
        default:
            FAIL();
    }
    const float * rptr = result.get();
    for (int i=0; i < param.channels * param.width * param.height; i++) {
        ASSERT_NEAR(rptr[i], expect, 0.5f);
    }
}

TEST_P(ParamSingletonLayerTest, SingletonTestDeep) {
    auto param = GetParam();
    gpu::SingletonArithLayerBuilder bld("single", param.oper);
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels).type(LayerType::SINGLETON_ARITH);
    bld.deep().operand(param.operand2);
    gpu::SingletonArithmeticLayer layer(bld, 1);
    std::unique_ptr<float[]> input(generateConstantData(param.operand1, param.channels, param.width, param.height));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    float expect = 0.f;
    switch (param.oper) {
        case ArithType::ADD:
            expect = param.operand1 + param.operand2;
            break;
        case ArithType::SUB:
            expect = param.operand1 - param.operand2;
            break;
        case ArithType::MUL:
            expect = param.operand1 * param.operand2;
            break;
        case ArithType::DIV:
            expect = param.operand1 / param.operand2;
            break;
        default:
            FAIL();
    }
    const float * rptr = result.get();
    for (int i=0; i < param.channels * param.width * param.height; i++) {
        ASSERT_NEAR(rptr[i], expect, 0.5f);
    }
}

TEST_P(ParamArithLayerTest, ArithTestShallow) {
    auto param = GetParam();
    gpu::GPULayerBuilder bld("arith");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels).type(param.oper);
    gpu::AddSubLayer layer(bld, 1);
    std::unique_ptr<float[]> input1(generateConstantData(param.operand1, param.channels, param.width, param.height));
    std::unique_ptr<float[]> input2(generateConstantData(param.operand2, param.channels, param.width, param.height));
    ASSERT_NE(input1, nullptr);
    ASSERT_NE(input2, nullptr);
    std::vector<const float *> inputs{input1.get(), input2.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    float expect = 0.f;
    switch (param.oper) {
        case LayerType::ADD:
            expect = param.operand1 + param.operand2;
            break;
        case LayerType::SUB:
            expect = param.operand1 - param.operand2;
            break;
        default:
            FAIL();
    }
    const float * rptr = result.get();
    for (int i=0; i < param.channels * param.width * param.height; i++) {
        ASSERT_NEAR(rptr[i], expect, 0.5f);
    }
}

// TODO (mw) more test patterns, maybe fuzz-testing with randomization

INSTANTIATE_TEST_CASE_P(SingleAdd, ParamSingletonLayerTest, testing::Values(
                                                                SingleArithParam(3.0f   , 30.0f,ArithType::ADD,400,300, 4),
                                                                SingleArithParam(-2.0f  ,  1.0f,ArithType::ADD,200,200, 5),
                                                                SingleArithParam(10.0f  , -10.f,ArithType::ADD, 16, 16,40),
                                                                SingleArithParam(-100.0f, 23.0f,ArithType::ADD, 55, 57,30),
                                                                SingleArithParam(15.0f  ,-16.0f,ArithType::ADD, 99, 52,47)));

INSTANTIATE_TEST_CASE_P(SingleSubtraction, ParamSingletonLayerTest, testing::Values(
                                                                SingleArithParam(3.0f   , 30.0f,ArithType::SUB,400,300, 4),
                                                                SingleArithParam(-2.0f  ,  1.0f,ArithType::SUB,200,200, 5),
                                                                SingleArithParam(10.0f  , -10.f,ArithType::SUB, 16, 16,40),
                                                                SingleArithParam(-100.0f, 23.0f,ArithType::SUB, 55, 57,30),
                                                                SingleArithParam(15.0f  ,-16.0f,ArithType::SUB, 99, 52,47)));

INSTANTIATE_TEST_CASE_P(SingleMultiplication, ParamSingletonLayerTest, testing::Values(
                                                                SingleArithParam(3.0f   , 30.0f,ArithType::MUL,400,300, 4),
                                                                SingleArithParam(-2.0f  ,  1.0f,ArithType::MUL,200,200, 5),
                                                                SingleArithParam(10.0f  , -10.f,ArithType::MUL, 16, 16,40),
                                                                SingleArithParam(-100.0f, 23.0f,ArithType::MUL, 55, 57,30),
                                                                SingleArithParam(15.0f  ,-16.0f,ArithType::MUL, 99, 52,47)));

INSTANTIATE_TEST_CASE_P(SingleDivision, ParamSingletonLayerTest, testing::Values(
                                                                SingleArithParam(3.0f   , 30.0f,ArithType::DIV,400,300, 4),
                                                                SingleArithParam(-2.0f  ,  1.0f,ArithType::DIV,200,200, 5),
                                                                SingleArithParam(10.0f  , -10.f,ArithType::DIV, 16, 16,40),
                                                                SingleArithParam(-100.0f, 23.0f,ArithType::DIV, 55, 57,30),
                                                                SingleArithParam(15.0f  ,-16.0f,ArithType::DIV, 99, 52,47)));

INSTANTIATE_TEST_CASE_P(Addition, ParamArithLayerTest, testing::Values(
                                                                ArithParam(3.0f   , 30.0f,LayerType::ADD,400,300, 4),
                                                                ArithParam(-2.0f  ,  1.0f,LayerType::ADD,200,200, 5),
                                                                ArithParam(10.0f  , -10.f,LayerType::ADD, 16, 16,40),
                                                                ArithParam(-100.0f, 23.0f,LayerType::ADD, 55, 57,30),
                                                                ArithParam(15.0f  ,-16.0f,LayerType::ADD, 99, 52,47)));

INSTANTIATE_TEST_CASE_P(Subtraction, ParamArithLayerTest, testing::Values(
                                                                ArithParam(3.0f   , 30.0f,LayerType::SUB,400,300, 4),
                                                                ArithParam(-2.0f  ,  1.0f,LayerType::SUB,200,200, 5),
                                                                ArithParam(10.0f  , -10.f,LayerType::SUB, 16, 16,40),
                                                                ArithParam(-100.0f, 23.0f,LayerType::SUB, 55, 57,30),
                                                                ArithParam(15.0f  ,-16.0f,LayerType::SUB, 99, 52,47)));

// vim: set expandtab ts=4 sw=4:
