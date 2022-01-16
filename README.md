# 初赛

## 记录

1. PGD12d+2PGD12r+3PGD12d+light+wasserstein: 98.0788
2. PGD12d+2PGD12r+3PGD12d+gaussian+wasserstein: 97.5600
3. PGD12d+2PGD12r+3PGD12d+3PGD12r+wasserstein: 97.5082
4. PGD12d+2PGD12r+3PGD12d+salt+wasserstein: 97.3802
5. PGD12d+2PGD12r+3PGD12r+light+wasserstein: 98.2233
6. PGD12r+2PGD12r+3PGD12r+light+wasserstein: 97.2687
7. PGD12d+2PGD12r+3PGD12r+light+wasserstein_init0.22: 98.0775
8. PGD12d+2PGD12r+3PGD12r+light_85-95+wasserstein: 98.1613
9. PGD12d+2PGD12r+3PGD12r+light_PGD2r+wasserstein: 97.7342
10. PGD12d+2PGD12r+3PGD12r+light_78-85+wasserstein: 98.0922
11. PGD12d+2PGD12r+3PGD12r+light+wasserstein+softlabel: 97.9325
12. PGD12d+2PGD12r+test+light+wasserstein: 97.8097
12. PGD12d+2PGD12r+MI-DI-FGSM+light+wasserstein: 96.7933

## 总结

densenet121比resnet50鲁棒

numpy数组转化为uint8类型前记得round

# 复赛
## 记录
[模型测试记录](https://docs.qq.com/sheet/DTXZHZWhRSGFpYUps?tab=BB08J2)
1. 5PGD-d
2. rrrdd
3. _rrdd
4. ___rd
5. ___rr
6. ____d
7. ____r

## 总结

+ Bilateral Adversarial Training：

    1.用更强的攻击来对抗训练不一定会得到更鲁棒的模型

    2.梯度幅度小的模型相对会更鲁棒

+ soft label是否需要保持一致性？利用鲁棒模型蒸馏

1w clean + 1w 类PGD + 1w wasserstein + 1w color +