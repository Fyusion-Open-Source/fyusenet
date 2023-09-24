//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// <source description>
// Creator:
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "resnet_provider.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor around existing memory block
 *
 * @param memory Pointer to memory that holds weight data
 * @param bytes Total size of memory block (in bytes)
 */
ResNet50Provider::ResNet50Provider(const uint8_t *memory, size_t bytes) : ResNet50Provider() {
    if (bytes < totalWeightBytes_) throw std::runtime_error("Insufficient weight data supplied");
    assert(wbData_);
    memcpy(wbData_, memory, totalWeightBytes_);
}


/**
 * @brief Constructor around file
 *
 * @param fileName File to load
 */
ResNet50Provider::ResNet50Provider(const std::string& fileName) : ResNet50Provider() {
    assert(wbData_);
    loadFile(fileName, (int)(totalWeightBytes_ / sizeof(float)));
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * Initializes the provider with the necessary information about where to find the weights in the
 * supplied memory block or file.
 */
ResNet50Provider::ResNet50Provider() : fyusion::fyusenet::ParameterProvider() {
    totalWeightBytes_ = 102304184;
    wbData_ = new float[totalWeightBytes_ / sizeof(float)];
    weightBlocks_.emplace(2, wrapper(wbData_ + 0));         // BN
    weightBlocks_.emplace(3, wrapper(wbData_ + 70));
    biasBlocks_.emplace(3, wrapper(wbData_ + 6));
    bnBlocks_.emplace(3, wrapper(wbData_ + 9478));
    weightBlocks_.emplace(5, wrapper(wbData_ + 9606));      // BN
    weightBlocks_.emplace(6, wrapper(wbData_ + 9798));
    biasBlocks_.emplace(6, wrapper(wbData_ + 9734));
    bnBlocks_.emplace(6, wrapper(wbData_ + 13894));
    weightBlocks_.emplace(7, wrapper(wbData_ + 67974));
    biasBlocks_.emplace(7, wrapper(wbData_ + 67718));
    weightBlocks_.emplace(8, wrapper(wbData_ + 14086));
    biasBlocks_.emplace(8, wrapper(wbData_ + 14022));
    bnBlocks_.emplace(8, wrapper(wbData_ + 50950));
    weightBlocks_.emplace(9, wrapper(wbData_ + 51334));
    biasBlocks_.emplace(9, wrapper(wbData_ + 51078));
    weightBlocks_.emplace(10, wrapper(wbData_ + 84358));    // BN
    weightBlocks_.emplace(11, wrapper(wbData_ + 84934));
    biasBlocks_.emplace(11, wrapper(wbData_ + 84870));
    bnBlocks_.emplace(11, wrapper(wbData_ + 101318));
    weightBlocks_.emplace(12, wrapper(wbData_ + 101510));
    biasBlocks_.emplace(12, wrapper(wbData_ + 101446));
    bnBlocks_.emplace(12, wrapper(wbData_ + 138374));
    weightBlocks_.emplace(13, wrapper(wbData_ + 138758));
    biasBlocks_.emplace(13, wrapper(wbData_ + 138502));
    weightBlocks_.emplace(14, wrapper(wbData_ + 155142));   // BN
    weightBlocks_.emplace(15, wrapper(wbData_ + 155718));
    biasBlocks_.emplace(15, wrapper(wbData_ + 155654));
    bnBlocks_.emplace(15, wrapper(wbData_ + 172102));
    weightBlocks_.emplace(16, wrapper(wbData_ + 172294));
    biasBlocks_.emplace(16, wrapper(wbData_ + 172230));
    bnBlocks_.emplace(16, wrapper(wbData_ + 209158));
    weightBlocks_.emplace(17, wrapper(wbData_ + 209542));
    biasBlocks_.emplace(17, wrapper(wbData_ + 209286));
    bnBlocks_.emplace(17, wrapper(wbData_ + 225926));
    weightBlocks_.emplace(18, wrapper(wbData_ + 226566));
    biasBlocks_.emplace(18, wrapper(wbData_ + 226438));
    bnBlocks_.emplace(18, wrapper(wbData_ + 259334));
    weightBlocks_.emplace(19, wrapper(wbData_ + 473990));
    biasBlocks_.emplace(19, wrapper(wbData_ + 473478));
    weightBlocks_.emplace(20, wrapper(wbData_ + 259718));
    biasBlocks_.emplace(20, wrapper(wbData_ + 259590));
    bnBlocks_.emplace(20, wrapper(wbData_ + 407174));
    weightBlocks_.emplace(21, wrapper(wbData_ + 407942));
    biasBlocks_.emplace(21, wrapper(wbData_ + 407430));
    weightBlocks_.emplace(22, wrapper(wbData_ + 605062));   // BN
    weightBlocks_.emplace(23, wrapper(wbData_ + 606214));
    biasBlocks_.emplace(23, wrapper(wbData_ + 606086));
    bnBlocks_.emplace(23, wrapper(wbData_ + 671750));
    weightBlocks_.emplace(24, wrapper(wbData_ + 672134));
    biasBlocks_.emplace(24, wrapper(wbData_ + 672006));
    bnBlocks_.emplace(24, wrapper(wbData_ + 819590));
    weightBlocks_.emplace(25, wrapper(wbData_ + 820358));
    biasBlocks_.emplace(25, wrapper(wbData_ + 819846));
    weightBlocks_.emplace(26, wrapper(wbData_ + 885894));   // BN
    weightBlocks_.emplace(27, wrapper(wbData_ + 887046));
    biasBlocks_.emplace(27, wrapper(wbData_ + 886918));
    bnBlocks_.emplace(27, wrapper(wbData_ + 952582));
    weightBlocks_.emplace(28, wrapper(wbData_ + 952966));
    biasBlocks_.emplace(28, wrapper(wbData_ + 952838));
    bnBlocks_.emplace(28, wrapper(wbData_ + 1100422));
    weightBlocks_.emplace(29, wrapper(wbData_ + 1101190));
    biasBlocks_.emplace(29, wrapper(wbData_ + 1100678));
    weightBlocks_.emplace(30, wrapper(wbData_ + 1166726));  // BN
    weightBlocks_.emplace(31, wrapper(wbData_ + 1167878));
    biasBlocks_.emplace(31, wrapper(wbData_ + 1167750));
    bnBlocks_.emplace(31, wrapper(wbData_ + 1233414));
    weightBlocks_.emplace(32, wrapper(wbData_ + 1233798));
    biasBlocks_.emplace(32, wrapper(wbData_ + 1233670));
    bnBlocks_.emplace(32, wrapper(wbData_ + 1381254));
    weightBlocks_.emplace(33, wrapper(wbData_ + 1382022));
    biasBlocks_.emplace(33, wrapper(wbData_ + 1381510));
    bnBlocks_.emplace(33, wrapper(wbData_ + 1447558));
    weightBlocks_.emplace(34, wrapper(wbData_ + 1448838));
    biasBlocks_.emplace(34, wrapper(wbData_ + 1448582));
    bnBlocks_.emplace(34, wrapper(wbData_ + 1579910));
    weightBlocks_.emplace(35, wrapper(wbData_ + 2435206));
    biasBlocks_.emplace(35, wrapper(wbData_ + 2434182));
    weightBlocks_.emplace(36, wrapper(wbData_ + 1580678));
    biasBlocks_.emplace(36, wrapper(wbData_ + 1580422));
    bnBlocks_.emplace(36, wrapper(wbData_ + 2170502));
    weightBlocks_.emplace(37, wrapper(wbData_ + 2172038));
    biasBlocks_.emplace(37, wrapper(wbData_ + 2171014));
    weightBlocks_.emplace(38, wrapper(wbData_ + 2959494));  // BN
    weightBlocks_.emplace(39, wrapper(wbData_ + 2961798));
    biasBlocks_.emplace(39, wrapper(wbData_ + 2961542));
    bnBlocks_.emplace(39, wrapper(wbData_ + 3223942));
    weightBlocks_.emplace(40, wrapper(wbData_ + 3224710));
    biasBlocks_.emplace(40, wrapper(wbData_ + 3224454));
    bnBlocks_.emplace(40, wrapper(wbData_ + 3814534));
    weightBlocks_.emplace(41, wrapper(wbData_ + 3816070));
    biasBlocks_.emplace(41, wrapper(wbData_ + 3815046));
    weightBlocks_.emplace(42, wrapper(wbData_ + 4078214));  // BN
    weightBlocks_.emplace(43, wrapper(wbData_ + 4080518));
    biasBlocks_.emplace(43, wrapper(wbData_ + 4080262));
    bnBlocks_.emplace(43, wrapper(wbData_ + 4342662));
    weightBlocks_.emplace(44, wrapper(wbData_ + 4343430));
    biasBlocks_.emplace(44, wrapper(wbData_ + 4343174));
    bnBlocks_.emplace(44, wrapper(wbData_ + 4933254));
    weightBlocks_.emplace(45, wrapper(wbData_ + 4934790));
    biasBlocks_.emplace(45, wrapper(wbData_ + 4933766));
    weightBlocks_.emplace(46, wrapper(wbData_ + 5196934));  // BN
    weightBlocks_.emplace(47, wrapper(wbData_ + 5199238));
    biasBlocks_.emplace(47, wrapper(wbData_ + 5198982));
    bnBlocks_.emplace(47, wrapper(wbData_ + 5461382));
    weightBlocks_.emplace(48, wrapper(wbData_ + 5462150));
    biasBlocks_.emplace(48, wrapper(wbData_ + 5461894));
    bnBlocks_.emplace(48, wrapper(wbData_ + 6051974));
    weightBlocks_.emplace(49, wrapper(wbData_ + 6053510));
    biasBlocks_.emplace(49, wrapper(wbData_ + 6052486));
    weightBlocks_.emplace(50, wrapper(wbData_ + 6315654));  // BN
    weightBlocks_.emplace(51, wrapper(wbData_ + 6317958));
    biasBlocks_.emplace(51, wrapper(wbData_ + 6317702));
    bnBlocks_.emplace(51, wrapper(wbData_ + 6580102));
    weightBlocks_.emplace(52, wrapper(wbData_ + 6580870));
    biasBlocks_.emplace(52, wrapper(wbData_ + 6580614));
    bnBlocks_.emplace(52, wrapper(wbData_ + 7170694));
    weightBlocks_.emplace(53, wrapper(wbData_ + 7172230));
    biasBlocks_.emplace(53, wrapper(wbData_ + 7171206));
    weightBlocks_.emplace(54, wrapper(wbData_ + 7434374));  // BN
    weightBlocks_.emplace(55, wrapper(wbData_ + 7436678));
    biasBlocks_.emplace(55, wrapper(wbData_ + 7436422));
    bnBlocks_.emplace(55, wrapper(wbData_ + 7698822));
    weightBlocks_.emplace(56, wrapper(wbData_ + 7699590));
    biasBlocks_.emplace(56, wrapper(wbData_ + 7699334));
    bnBlocks_.emplace(56, wrapper(wbData_ + 8289414));
    weightBlocks_.emplace(57, wrapper(wbData_ + 8290950));
    biasBlocks_.emplace(57, wrapper(wbData_ + 8289926));
    bnBlocks_.emplace(57, wrapper(wbData_ + 8553094));
    weightBlocks_.emplace(58, wrapper(wbData_ + 8555654));
    biasBlocks_.emplace(58, wrapper(wbData_ + 8555142));
    bnBlocks_.emplace(58, wrapper(wbData_ + 9079942));
    weightBlocks_.emplace(59, wrapper(wbData_ + 12494470));
    biasBlocks_.emplace(59, wrapper(wbData_ + 12492422));
    weightBlocks_.emplace(60, wrapper(wbData_ + 9081478));
    biasBlocks_.emplace(60, wrapper(wbData_ + 9080966));
    bnBlocks_.emplace(60, wrapper(wbData_ + 11440774));
    weightBlocks_.emplace(61, wrapper(wbData_ + 11443846));
    biasBlocks_.emplace(61, wrapper(wbData_ + 11441798));
    weightBlocks_.emplace(62, wrapper(wbData_ + 14591622));  // BN
    weightBlocks_.emplace(63, wrapper(wbData_ + 14596230));
    biasBlocks_.emplace(63, wrapper(wbData_ + 14595718));
    bnBlocks_.emplace(63, wrapper(wbData_ + 15644806));
    weightBlocks_.emplace(64, wrapper(wbData_ + 15646342));
    biasBlocks_.emplace(64, wrapper(wbData_ + 15645830));
    bnBlocks_.emplace(64, wrapper(wbData_ + 18005638));
    weightBlocks_.emplace(65, wrapper(wbData_ + 18008710));
    biasBlocks_.emplace(65, wrapper(wbData_ + 18006662));
    weightBlocks_.emplace(66, wrapper(wbData_ + 19057286));  // BN
    weightBlocks_.emplace(67, wrapper(wbData_ + 19061894));
    biasBlocks_.emplace(67, wrapper(wbData_ + 19061382));
    bnBlocks_.emplace(67, wrapper(wbData_ + 20110470));
    weightBlocks_.emplace(68, wrapper(wbData_ + 20112006));
    biasBlocks_.emplace(68, wrapper(wbData_ + 20111494));
    bnBlocks_.emplace(68, wrapper(wbData_ + 22471302));
    weightBlocks_.emplace(69, wrapper(wbData_ + 22474374));
    biasBlocks_.emplace(69, wrapper(wbData_ + 22472326));
    bnBlocks_.emplace(69, wrapper(wbData_ + 23522950));
    weightBlocks_.emplace(72, wrapper(wbData_ + 23528046));
    biasBlocks_.emplace(72, wrapper(wbData_ + 23527046));
}


/**
 * @brief Load weights from file
 *
 * @param fileName File to load
 * @param numFloats Number of 32-bit floats in that file
 *
 * Loads the weights from the specified file into the internal memory block.
 */
void ResNet50Provider::loadFile(const std::string& fileName, int numFloats) {
    assert(wbData_);
    FILE * f = fopen(fileName.c_str(), "rb");
    if (f) {
        size_t read = fread(wbData_, sizeof(float), numFloats, f);
        fclose(f);
        if (read != (size_t)numFloats) {
            throw std::runtime_error("Insufficient weight data supplied in file " + fileName);
        }
    } else {
        throw std::runtime_error("Cannot open file " + fileName);
    };

}

// vim: set expandtab ts=4 sw=4:

