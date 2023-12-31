# AddBlockCust

## Overview

This sample describes the implementation of the AI CPU custom operator AddBlockCust to illustrate how the customized AI CPU operator supports block-based parallel computing (Enable multi-core parallel computing).

1.  Conditions for enabling multi-core parallel computing.
    -   In the calculation process, data is not associated with each other, and can be divided into multiple data blocks for independent calculation. If there is data dependency between input parameters, block parallel calculation cannot be performed.
    -   The location and size of the results generated by each piece of data can be predicted before calculation. If it cannot be predicted, the calculation results need to be spliced. The splicing process will bring additional performance losses.

2.  How to enable multi-core parallel computing。
    -   When using this function, you need to set opInfo. flagSupportBlockDim=True in the operator information library definition, and set opInfo. functionName=RunCpuKernelWithBlock。
    -   OpInfo.blockDimByIndex This field represents segmentation based on a dimension of the first input parameter shape. The default value is - 1。

The AddBlockCust operator returns the sum of its operands, as shown in the following figure.

**Figure  1**  Add operator function diagram<a name="en-us_topic_0229823836_fig1134425318216"></a>  
![](https://images.gitee.com/uploads/images/2021/0114/162517_9f334f35_5474059.png "add-operator-function-diagram.png")


## Operator Analysis

1.  The mathematical expression of the AddBlockCust operator is as follows:

    ```
     z=x+y
    ```

    The Add operator adds two input parameters to obtain the final result  **z**  and return it.

2.  Specify the inputs and output.
    -   The AddBlockCust operator has two inputs,  **x**  and  **y**, and outputs the result  **z**.
    -   The supported input data types include float32, int32, and int64. The output has the same data type as the inputs.
    -   The operator input supports all shapes. The output has the same shape as the inputs.
    -   The operator input supports the following formats:  **NCHW**,  **NC1HWC0**,  **NHWC**, and  **ND**.

3.  Specify the operator implementation file name and  _OpType_.

    -   Name  _**OpType**_  in upper camel case and indicate the separation of words with a single capitalized letter.
    -   Name the operator implementation file after  **_OpType_**  as follows:
        -   Replace the first uppercase letter with a lowercase letter.

            Example: Abc -\> abc

        -   Replace each uppercase letter following lowercase letters with an underscore \(\_\) and a lowercase letter.

            Example: AbcDef -\> abc\_def

        -   Uppercase letters following a digit or an uppercase letter are regarded as a character string. If there is a lowercase letter after this string, replace the last uppercase letter in this string with an underscore \(\_\) and a lowercase letter, and replace the other uppercase letters with lowercase letters. If there is no lowercase letter after the string, directly replace the string with lowercase letters.

            Examples: ABCDef -\> abc\_def; Abc2DEf -\> abc2d\_ef; Abc2DEF -\> abc2def; ABC2dEF -\> abc2d\_ef



    To avoid messing up with the built-in Reshape operators, define the  _OpType_  as  **ReshapeCust**  and the implementation file name as  **reshape\_cust**.
    In this example,  _OpType_  of the operator is defined as  **AddBlockCust**, the implementation file name is defined as  **add\_block\_cust**.


## Code Implementation

-   Operator Implementation

    For details about the implementation code of the AddBlockCust operator, see  [add\_block\_cust\_kernels.h](../cpukernel/impl/add_block_cust_kernels.h) and [add\_block\_cust\_kernels.cc](../cpukernel/impl/add_block_cust_kernels.cc).
    This operator supports block-based parallel calculation. For details about the calculation logic, see the AddComputeWithBlock function in [add\_block\_cust\_kernels.cc](../cpukernel/impl/add_block_cust_kernels.cc).

-   Operator Prototype Definition

    The key point of prototype definition is inferring the shape of the output tensor.

     The principle of inferring the output shape is as follows: Obtain the two input shapes, broadcast them to the same shape, and assign the larger value of each dimension of the two inputs to form the output shape. For details about the code implementation of the  **InferShapeAndTypeAddBlock**  function, see  [add_block_cust.cc](../op_proto/add_block_cust.cc).

-   Operator Information Library

    For details about the operator information library of AddBlockCust, see  [add\_block\_cust.ini](../cpukernel/op_info_cfg/aicpu_kernel/add_block_cust.ini).


## Supported SoCs

All