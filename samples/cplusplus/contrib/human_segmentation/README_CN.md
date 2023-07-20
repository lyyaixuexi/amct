中文|[English](README.md)

**本样例为大家学习昇腾软件栈提供参考，非商业目的！**

**本README只提供命令行方式运行样例的指导，如需在Mindstudio下运行样例，请参考[Mindstudio运行视频样例wiki](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E8%A7%86%E9%A2%91%E6%A0%B7%E4%BE%8B?sort_id=3170138)。**

## 视频人体语义分割样例
功能：使用语义分割模型对输入的视频中人体进行语义分割推理。    
样例输入：包含人体的mp4视频。    
样例输出：presenter界面展现推理结果。    

### 前置条件
请检查以下条件要求是否满足，如不满足请按照备注进行相应处理。如果CANN版本升级，请同步检查第三方依赖是否需要重新安装（5.0.4及以上版本第三方依赖和5.0.4以下版本有差异，需要重新安装）。
| 条件 | 要求 | 备注 |
|---|---|---|
| CANN版本 | >=5.0.4 | 请参考CANN样例仓介绍中的[安装步骤](https://gitee.com/ascend/samples#%E5%AE%89%E8%A3%85)完成CANN安装，如果CANN低于要求版本请根据[版本说明](https://gitee.com/ascend/samples/blob/master/README_CN.md#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)切换samples仓到对应CANN版本 |
| 硬件要求 | Atlas200DK/Atlas300([ai1s](https://support.huaweicloud.com/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366))  | 当前已在Atlas200DK和Atlas300测试通过，产品说明请参考[硬件平台](https://ascend.huawei.com/zh/#/hardware/product) ，其他产品可能需要另做适配|
| 第三方依赖 | presentagent, opencv, ffmpeg+acllite | 请参考[第三方依赖安装指导（C++样例）](../../environment)完成对应安装 |

### 样例准备
1. 获取源码包。    
   可以使用以下两种方式下载，请选择其中一种进行源码准备。   
    - 命令行方式下载（下载时间较长，但步骤简单）。
       ```    
       # 开发环境，非root用户命令行中执行以下命令下载源码仓。    
       cd ${HOME}     
       git clone https://gitee.com/ascend/samples.git
       ```
       **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
       ```
       git checkout v0.5.0
       ```   
    - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。   
       **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**   
       ``` 
        # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。    
        # 2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/ascend-samples-master.zip】。     
        # 3. 开发环境中，执行以下命令，解压zip包。     
        cd ${HOME}    
        unzip ascend-samples-master.zip
        ```

2. 模型转换。     
    |  **模型名称**  |  **模型说明**  |  **模型下载路径**  |
    |---|---|---|
    |  human_segmentation| 人体语义分割推理模型。是基于Tensorflow的语义分割模型。  |  请参考[https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/human_segmentation/ATC_human_segmentation_tf_AE](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/human_segmentation/ATC_human_segmentation_tf_AE)目录中README.md下载原始模型章节下载模型和权重文件配置文件。 |

    ```
    # 为了方便下载，在这里直接给出原始模型下载及模型转换命令,可以直接拷贝执行。也可以参照上表在modelzoo中下载并手工转换，以了解更多细节。     
    
    cd $HOME/samples/cplusplus/contrib/human_segmentation/model    
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/human_segmentation.pb   
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/insert_op.cfg
    atc --input_shape="input_rgb:1,512,512,3" --input_format=NHWC --output=human_segmentation --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=3 --model=./human_segmentation.pb
    ```


### 样例部署
执行以下命令，执行编译脚本，开始样例编译。     
```
cd  $HOME/samples/cplusplus/contrib/human_segmentation/scripts    
bash sample_build.sh
```

### 样例运行
1. 执行以下命令,将开发环境的 **human_segmentation** 目录上传到运行环境中，例如 **/home/HwHiAiUser**，并以HwHiAiUser（运行用户）登录运行环境（Host）。      
   ```
   # 【xxx.xxx.xxx.xxx】为运行环境ip，200DK在USB连接时一般为192.168.1.2，300（ai1s）为对应的公网ip。
   scp -r $HOME/samples/cplusplus/contrib/human_segmentation HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser    
   ssh HwHiAiUser@xxx.xxx.xxx.xxx     
   cd $HOME/samples/cplusplus/contrib/human_segmentation/scripts
   ```

2. <a name="step_2"></a>执行运行脚本，开始样例运行。          
   ```
   bash sample_run.sh
   ```

### 查看结果
1. 打开presentserver网页界面。   
   - 使用产品为200DK开发者板。    
     打开启动Presenter Server服务时提示的URL即可。   
   - 使用产品为300加速卡（ai1s云端推理环境）。    
     **以300加速卡（ai1s）内网ip为192.168.0.194，公网ip为124.70.8.192举例说明。**     
     启动Presenter Server服务时提示为Please visit http://192.168.0.194:7009 for display server。    
     只需要将URL中的内网ip：192.168.0.194替换为公网ip：124.70.8.192，则URL为 http://124.70.8.192:7009。    
     然后在windows下的浏览器中打开URL即可。    
2. 等待Presenter Agent传输数据给服务端，单击“Refresh“刷新，当有数据时相应的Channel 的Status变成绿色。    
3. 单击右侧对应的View Name链接，查看结果。    

### 常见错误
请参考[常见问题定位](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D)对遇到的错误进行排查。如果wiki中不包含，请在samples仓提issue反馈。