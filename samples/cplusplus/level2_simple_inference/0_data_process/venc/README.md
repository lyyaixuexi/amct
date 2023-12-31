English|[中文](README_CN.md)

**This sample provides reference for you to learn the Ascend AI Software Stack and cannot be used for commercial purposes.**

**This README file provides only guidance for running the sample in command line (CLI) mode. For details about how to run the sample in MindStudio, see [Running Video Samples in MindStudio](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E5%9B%BE%E7%89%87%E6%A0%B7%E4%BE%8B?sort_id=3164874).**

## VENC Sample
Function: Encode videos by calling the **venc** API of DVPP.   
Input: source YUV image.   
Output: encoded H.264 file.  

### Prerequisites
Check whether the following requirements are met. If not, perform operations according to the remarks. If the CANN version is upgraded, check whether the third-party dependencies need to be reinstalled. (The third-party dependencies for 5.0.4 and later versions are different from those for earlier versions.)
| Item| Requirement| Remarks|
|---|---|---|
| CANN version| ≥ 5.0.4| Install the CANN by referring to [Sample Deployment](https://gitee.com/ascend/samples#%E5%AE%89%E8%A3%85) in the *About Ascend Samples Repository*. If the CANN version is earlier than the required version, switch to the **samples** repository specific to the CANN version. See [Release Notes](https://gitee.com/ascend/samples/blob/master/README.md).|
| Hardware| Atlas200DK/Atlas300 ([AI1s](https://support.huaweicloud.com/en-us/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366))  | Currently, the Atlas 200 DK and Atlas 300 have passed the test. For details about the product description, see [Hardware Platform](https://ascend.huawei.com/en/#/hardware/product). For other products, adaptation may be required.|
| Third-party dependency| Installation preparation| Set environment variables based on the [installation preparation](.../../environment) of each third-party dependency.|

### Software Preparation

You can obtain the source package in either of the following ways:  
  - Command line (The download takes a long time, but the procedure is simple.)
     ```    
     # In the development environment, run the following commands as a non-root user to download the source repository:   
     cd ${HOME}     
     git clone https://gitee.com/ascend/samples.git
     ```
     **Note: To switch to another tag (for example, v0.5.0), run the following command:**
     ```
     git checkout v0.5.0
     ```
  - Compressed package (The download takes a short time, but the procedure is complex.)  
     **Note: If you want to download the code of another version, switch the branch of the samples repository according to the prerequisites.**   
     ``` 
      # 1. Click Clone or Download in the upper right corner of the samples repository and click Download ZIP.   
      # 2. Upload the .zip package to the home directory of a common user in the development environment, for example, ${HOME}/ascend-samples-master.zip.    
      # 3. In the development environment, run the following commands to unzip the package:    
      cd ${HOME}    
      unzip ascend-samples-master.zip
     ```


### Sample Deployment
Run the following commands to execute the compilation script to start sample compilation:  
```
cd ${HOME}/samples/cplusplus/level2_simple_inference/0_data_process/venc/scripts    
bash sample_build.sh
```

### Sample Running

**Note: If the development environment and operating environment are set up on the same server, skip step 1 and go to [step 2](#step_2) directly.**  

1. Run the following commands to upload the **venc** directory in the development environment to any directory in the operating environment, for example, **/home/HwHiAiUser**, and log in to the operating environment (host) as the running user (**HwHiAiUser**):
    ```
    # In the following information, xxx.xxx.xxx.xxx is the IP address of the operating environment. The IP address of Atlas 200 DK is 192.168.1.2 when it is connected over the USB port, and that of Atlas 300 (AI1s) is the corresponding public IP address.
    scp -r ${HOME}/samples/cplusplus/level2_simple_inference/0_data_process/venc HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser    
    ssh HwHiAiUser@xxx.xxx.xxx.xxx     
    cd ${HOME}/venc/scripts
    ```

2. <a name="step_2"></a>Execute the script to run the sample.

    ```
    bash sample_run.sh
    ```

### Result Viewing
After the running is complete, the inference result is displayed in the CLI of the operating environment, and the encoded H.264 file is generated in **out/output** in the sample directory in the operating environment.

### Common Errors
For details about how to rectify the errors, see [Troubleshooting](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D). If an error is not included in Wiki, submit an issue to the **samples** repository.
